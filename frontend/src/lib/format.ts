const dateTime = new Intl.DateTimeFormat(undefined, {
  weekday: "short",
  day: "numeric",
  month: "short",
  hour: "numeric",
  minute: "2-digit",
});

const dayOnly = new Intl.DateTimeFormat(undefined, {
  weekday: "short",
  day: "numeric",
  month: "short",
});

export function formatDateTime(value: string): string {
  return dateTime.format(new Date(value));
}

/** Format a plain YYYY-MM-DD date, without letting a timezone move it a day. */
export function formatDay(value: string): string {
  return dayOnly.format(new Date(`${value}T12:00:00`));
}

export function isoDay(date: Date): string {
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${date.getFullYear()}-${month}-${day}`;
}

/** Return the seven days of the week containing `date`, Monday first. */
export function weekOf(date: Date): string[] {
  const monday = new Date(date);
  monday.setDate(date.getDate() - ((date.getDay() + 6) % 7));
  return Array.from({ length: 7 }, (_, offset) => {
    const day = new Date(monday);
    day.setDate(monday.getDate() + offset);
    return isoDay(day);
  });
}

export function shortTime(value: string): string {
  return value.slice(0, 5);
}

export function fileSize(bytes: number): string {
  return bytes < 1024 * 1024
    ? `${Math.ceil(bytes / 1024)} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
