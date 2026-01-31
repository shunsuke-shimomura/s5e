#[cfg(feature = "std")]
use chrono::{Datelike, NaiveDateTime, Timelike};

const DAY_OF_CENTURY: f64 = 36525.0;
const J2000_EPOCH_JD: f64 = 2451545.0;

#[derive(Debug, Clone)]
pub struct JulianDate {
    pub value: f64,
}

impl JulianDate {
    pub fn new(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
        nanosecond: u32,
    ) -> Self {
        let y = if month > 2 {
            year as f64
        } else {
            (year - 1) as f64
        };
        let m = if month > 2 {
            month as f64
        } else {
            (month + 12) as f64
        };

        let a = (y / 100.0).floor();
        let b = 2.0 - a + (a / 4.0).floor();

        let jd = (365.25 * (y + 4716.0)).floor() + (30.6001 * (m + 1.0)).floor() + day as f64 + b
            - 1524.5;

        // 時間の少数部分を追加
        let day_fraction = (hour as f64
            + (minute as f64 / 60.0)
            + ((second as f64 + nanosecond as f64 * 1.0e-9) / 3600.0))
            / 24.0;

        JulianDate {
            value: jd + day_fraction,
        }
    }
}

#[cfg(feature = "std")]
impl From<NaiveDateTime> for JulianDate {
    fn from(value: NaiveDateTime) -> Self {
        let year = value.year();
        let month = value.month();
        let day = value.day();
        let hour = value.hour();
        let minute = value.minute();
        let second = value.second();
        let nanosecond = value.nanosecond();
        JulianDate::new(year, month, day, hour, minute, second, nanosecond)
    }
}

pub struct Century {
    pub value: f64,
}

impl From<&JulianDate> for Century {
    fn from(jd: &JulianDate) -> Self {
        Century {
            value: (jd.value - J2000_EPOCH_JD) / DAY_OF_CENTURY,
        }
    }
}
