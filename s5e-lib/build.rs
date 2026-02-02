use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn download_file(url: &str, dest_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if Path::new(dest_path).exists() {
        println!(
            "File {} already exists, skipping download",
            dest_path.to_str().unwrap()
        );
        return Ok(());
    }

    println!("Downloading {} to {}", url, dest_path.to_str().unwrap());

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    let downloaded_bytes = rt.block_on(async { download_async(url, dest_path).await });

    println!("Download complete: {} bytes", downloaded_bytes);

    println!("Downloaded {} successfully", dest_path.to_str().unwrap());
    Ok(())
}

async fn download_async(url: &str, dest_path: &PathBuf) -> u64 {
    use futures_util::StreamExt;
    use std::io::Write;

    // HTTP client configuration
    let client = reqwest::Client::builder()
        // Connection timeout: 30 seconds
        .connect_timeout(std::time::Duration::from_secs(30))
        // Read timeout: 60 seconds (maximum wait time for next data chunk)
        .read_timeout(std::time::Duration::from_secs(60))
        // Do not set overall timeout
        .build()
        .expect("Failed to build HTTP client");

    // Send request
    let response = client
        .get(url)
        .send()
        .await
        .expect("Failed to start CSPICE download");

    // Check status code
    if !response.status().is_success() {
        panic!(
            "Failed to download CSPICE: HTTP {} from {}",
            response.status(),
            url
        );
    }

    // Get content size
    let total_size = response.content_length();
    if let Some(size) = total_size {
        println!(
            "Download size: {} bytes ({} MB)",
            size,
            size / (1024 * 1024)
        );
    }

    // Streaming download
    let mut file = std::fs::File::create(dest_path).expect("Failed to create download file");

    let mut downloaded = 0u64;
    let mut stream = response.bytes_stream();

    // Show initial progress
    println!("Starting download from {}", url);
    if let Some(total) = total_size {
        println!("Progress: 0 MB / {} MB (0%)", total / (1024 * 1024));
    }

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result.expect("Failed to read download chunk");

        file.write_all(&chunk)
            .expect("Failed to write to download file");

        downloaded += chunk.len() as u64;

        // Progress display (every 1MB)
        if downloaded % (1024 * 1024) < chunk.len() as u64 {
            if let Some(total) = total_size {
                let percent = (downloaded as f64 / total as f64 * 100.0) as u32;
                println!(
                    "Progress: {} MB / {} MB ({}%)",
                    downloaded / (1024 * 1024),
                    total / (1024 * 1024),
                    percent
                );
            } else {
                println!("Downloaded {} MB", downloaded / (1024 * 1024));
            }
        }
    }

    file.flush().expect("Failed to flush download file");

    // Verify download completion
    if let Some(total) = total_size
        && downloaded != total
    {
        panic!(
            "Download incomplete: got {} bytes, expected {} bytes",
            downloaded, total
        );
    }

    downloaded
}

fn main() {
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_dir = Path::new(&out_path).join("kernels");

    // Create kernels directory
    fs::create_dir_all(&kernel_dir).unwrap();

    // Define all required kernels based on S2E usage
    let kernels = vec![
        // Leap Seconds Kernel (LSK)
        (
            "latest_leapseconds.tls",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/latest_leapseconds.tls",
        ),
        // Planetary Constants Kernels (PCK)
        (
            "de-403-masses.tpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/de-403-masses.tpc",
        ),
        (
            "gm_de440.tpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc",
        ),
        (
            "pck00011.tpc",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc",
        ),
        // Spacecraft and Planet Kernels (SPK)
        (
            "de442s.bsp",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de442s.bsp",
        ),
        (
            "earth_latest_high_prec.bsp",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/earth_latest_high_prec.bpc",
        ),
    ];

    // Download all kernels
    for (filename, url) in kernels {
        let kernel_path = kernel_dir.join(filename);
        if let Err(e) = download_file(url, &kernel_path) {
            println!("cargo:error=Failed to download {}: {}", filename, e);
        }
    }

    println!("cargo:rerun-if-changed=./spice-kernel/teme.tf");
    // Embed ./spice-kernel/teme.tf into the kernel directory
    let project_root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let source_path = project_root.join("spice-kernel/teme.tf");
    let dest_path = kernel_dir.join("teme.tf");

    if let Err(e) = fs::copy(&source_path, &dest_path) {
        println!(
            "cargo:error=Failed to copy teme.tf to kernel directory: {}",
            e
        );
    } else {
        println!(
            "Successfully embedded teme.tf into kernel directory: {}",
            dest_path.display()
        );
    }

    // Pass kernel directory to the program
    println!("cargo:rustc-env=SPICE_KERNEL_DIR={}", kernel_dir.display());
}
