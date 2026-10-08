import type { ReactElement } from "react";
import type { Glyph as G } from "../cards";

// Hand-drawn 24x24 line icons for suits and characters (stroke = currentColor).
const PATHS: Record<G, ReactElement> = {
  chest: (
    <>
      <path d="M4 11a8 5.5 0 0 1 16 0" />
      <rect x="4" y="11" width="16" height="8.5" rx="1.2" />
      <path d="M4 14.5h16" />
      <rect x="10.4" y="12.8" width="3.2" height="3.6" rx=".6" fill="currentColor" stroke="none" />
    </>
  ),
  parrot: (
    <>
      <path d="M15.5 4.5c-4 0-7 3-7 7.5 0 2.6 1 4.8 2.6 6.5" />
      <path d="M15.5 4.5c2.3 0 4 1.6 4 3.7 0 1-.5 1.8-1.4 2.3l-2.6-1.1" />
      <path d="M8.5 12c-1.8 1.4-3 3.8-3 7.5 2.4-.3 4.3-1.2 5.6-1" />
      <circle cx="15.6" cy="7.4" r=".9" fill="currentColor" stroke="none" />
    </>
  ),
  map: (
    <>
      <path d="M3.5 6.5 9 4.5l6 2 5.5-2v13l-5.5 2-6-2-5.5 2z" />
      <path d="M9 4.5v13M15 6.5v13" strokeOpacity=".55" />
      <path d="m11 10.5 2 2m0-2-2 2" />
    </>
  ),
  flag: (
    <>
      <path d="M5.5 21V3.5" />
      <path d="M5.5 4h13l-2.6 4.2 2.6 4.3h-13" />
      <circle cx="11" cy="8.2" r="1.7" />
      <path d="M9.6 10.9l2.8-1.4M12.4 10.9 9.6 9.5" strokeOpacity=".7" />
    </>
  ),
  escape: (
    <>
      <path d="M6 21V3.5" />
      <path d="M6 4c3-1.6 5.5 1.6 9 0s3.5-.4 3.5-.4v8.6s-1.2-1-3.5 0-6 1.6-9 0" />
    </>
  ),
  mermaid: (
    <>
      <path d="M12 3.5c-1.5 3.5-1 6.8 0 9.5s.8 4.3-1.2 6" />
      <path d="M10.8 19c-1.6-.4-3.8.2-5.3 1.5 2.1.6 4.5.3 5.3-1.5Z" />
      <path d="M10.8 19c.9 1.6 3.1 2.3 5.3 1.9-1.2-1.5-3.5-2.3-5.3-1.9Z" />
      <path d="M8.5 8.5c1.2 1 2.4 1.3 3.5 1.1" strokeOpacity=".6" />
    </>
  ),
  pirate: (
    <>
      <path d="M5 4.5 17.5 17M19 4.5 6.5 17" />
      <path d="m15 19.5 4.5-4.5M9 19.5 4.5 15" />
      <path d="M4 3.5h2.5V6M20 3.5h-2.5V6" />
    </>
  ),
  tigress: (
    <>
      <path d="M7 4c1 5.5.6 10.5-2 16" />
      <path d="M12 3.5c.9 5.8.5 11-1.2 16.5" />
      <path d="M17 4c.6 5.5 0 10.5-2.6 16" />
    </>
  ),
  king: (
    <>
      <path d="M3.5 8.5 7.8 12l4.2-6.5 4.2 6.5 4.3-3.5-1.8 10H5.3z" />
      <path d="M5.3 18.5h13.4" />
      <circle cx="12" cy="4" r="1" fill="currentColor" stroke="none" />
    </>
  ),
};

export function Glyph({ name, size = 18, className }: { name: G; size?: number; className?: string }) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.6}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden
    >
      {PATHS[name]}
    </svg>
  );
}
