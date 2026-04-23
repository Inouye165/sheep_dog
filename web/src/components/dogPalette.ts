const DOG_PALETTE = ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#8b5cf6", "#ec4899"];

export function dogColor(index: number): string {
  return DOG_PALETTE[index % DOG_PALETTE.length];
}