/**
 * NSE Trading Day Utilities
 * 
 * Provides functions to calculate the next valid NSE trading day,
 * accounting for weekends and NSE holidays.
 * 
 * NSE is closed on:
 * - Saturdays (day 6)
 * - Sundays (day 0)
 * - NSE-specific holidays (listed below)
 */

// NSE Holidays for 2026 (from NSE official calendar)
// Format: 'YYYY-MM-DD'
const NSE_HOLIDAYS_2026: string[] = [
    '2026-01-26', // Republic Day
    '2026-03-10', // Maha Shivaratri
    '2026-03-17', // Holi
    '2026-03-30', // Id-ul-Fitr (Ramzan Eid)
    '2026-04-02', // Ram Navami
    '2026-04-03', // Good Friday
    '2026-04-14', // Dr. Ambedkar Jayanti
    '2026-04-21', // Mahavir Jayanti
    '2026-05-01', // Maharashtra Day
    '2026-06-06', // Id-ul-Adha (Bakri Eid)
    '2026-07-06', // Muharram
    '2026-08-15', // Independence Day
    '2026-08-26', // Janmashtami
    '2026-10-02', // Mahatma Gandhi Jayanti
    '2026-10-20', // Dussehra
    '2026-10-21', // Dussehra (2nd day)
    '2026-11-09', // Diwali Laxmi Pujan
    '2026-11-10', // Diwali Balipratipada
    '2026-11-12', // Guru Nanak Jayanti
    '2026-12-25', // Christmas
];

// Extended holiday list for 2025 (for backwards compatibility)
const NSE_HOLIDAYS_2025: string[] = [
    '2025-01-26', // Republic Day
    '2025-02-26', // Maha Shivaratri
    '2025-03-14', // Holi
    '2025-03-31', // Id-ul-Fitr
    '2025-04-06', // Ram Navami
    '2025-04-10', // Mahavir Jayanti
    '2025-04-14', // Dr. Ambedkar Jayanti
    '2025-04-18', // Good Friday
    '2025-05-01', // Maharashtra Day
    '2025-06-07', // Id-ul-Adha (Bakri Eid)
    '2025-08-15', // Independence Day
    '2025-08-16', // Janmashtami
    '2025-08-27', // Milad-un-Nabi
    '2025-10-02', // Mahatma Gandhi Jayanti
    '2025-10-21', // Dussehra
    '2025-10-22', // Diwali Laxmi Pujan
    '2025-11-05', // Guru Nanak Jayanti
    '2025-12-25', // Christmas
];

// Combined holiday set for quick lookup
const ALL_NSE_HOLIDAYS = new Set([...NSE_HOLIDAYS_2025, ...NSE_HOLIDAYS_2026]);

/**
 * Format a Date object to 'YYYY-MM-DD' string
 */
export function formatDateToISO(date: Date): string {
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    return `${year}-${month}-${day}`;
}

/**
 * Check if a given date is an NSE holiday
 */
export function isNSEHoliday(date: Date): boolean {
    const dateStr = formatDateToISO(date);
    return ALL_NSE_HOLIDAYS.has(dateStr);
}

/**
 * Check if a given date is a weekend (Saturday or Sunday)
 */
export function isWeekend(date: Date): boolean {
    const day = date.getDay();
    return day === 0 || day === 6; // Sunday = 0, Saturday = 6
}

/**
 * Check if NSE is open on a given date
 */
export function isNSETradingDay(date: Date): boolean {
    return !isWeekend(date) && !isNSEHoliday(date);
}

/**
 * Get the next valid NSE trading day from a given date
 * 
 * @param currentDate - The reference date (defaults to today)
 * @param includeCurrentDay - If true and currentDate is a trading day, return it
 * @returns The next valid NSE trading day
 * 
 * Examples:
 * - Friday → Returns Monday (if Monday is not a holiday)
 * - Saturday → Returns Monday (if Monday is not a holiday)
 * - Thursday before Good Friday → Returns next Monday
 */
export function getNextTradingDay(currentDate: Date = new Date(), includeCurrentDay: boolean = false): Date {
    // Clone the date to avoid mutating the input
    const nextDay = new Date(currentDate);

    // If not including current day, start from tomorrow
    if (!includeCurrentDay) {
        nextDay.setDate(nextDay.getDate() + 1);
    }

    // Keep advancing until we find a trading day
    // Safety limit of 30 days to prevent infinite loops
    let iterations = 0;
    const MAX_ITERATIONS = 30;

    while (!isNSETradingDay(nextDay) && iterations < MAX_ITERATIONS) {
        nextDay.setDate(nextDay.getDate() + 1);
        iterations++;
    }

    return nextDay;
}

/**
 * Format the next trading day for display
 * Returns a human-readable string like "Monday, Feb 10, 2026"
 */
export function formatNextTradingDay(date: Date): string {
    return date.toLocaleDateString('en-IN', {
        weekday: 'long',
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

/**
 * Get prediction target date information
 * Returns both the date object and formatted string
 */
export function getPredictionTargetDate(referenceDate: Date = new Date()): {
    date: Date;
    formatted: string;
    isoString: string;
    daysAway: number;
} {
    const targetDate = getNextTradingDay(referenceDate, false);
    const today = new Date(referenceDate);
    today.setHours(0, 0, 0, 0);
    const target = new Date(targetDate);
    target.setHours(0, 0, 0, 0);

    const daysAway = Math.round((target.getTime() - today.getTime()) / (1000 * 60 * 60 * 24));

    return {
        date: targetDate,
        formatted: formatNextTradingDay(targetDate),
        isoString: formatDateToISO(targetDate),
        daysAway
    };
}
