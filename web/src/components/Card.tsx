import { motion } from "motion/react";
import { cardName, glyphOf, isColored, tone, valueOf } from "../cards";
import { Glyph } from "./Glyph";

export interface CardProps {
  kind: number;
  size?: "lg" | "md" | "sm";
  /** strategy probability, shown as a gauge under the card */
  p?: number;
  best?: boolean;
  disabled?: boolean;
  selected?: boolean;
  badge?: string | number;
  onClick?: () => void;
  layoutId?: string;
}

const pct = (p: number) => (p >= 0.995 ? "100" : p < 0.005 ? "0" : (p * 100).toFixed(p < 0.1 ? 1 : 0));

export function Card({ kind, size = "md", p, best, disabled, selected, badge, onClick, layoutId }: CardProps) {
  const colored = isColored(kind);
  const label = colored ? String(valueOf(kind)) : "";
  const classes = ["card", `card-${size}`, tone(kind), best && "is-best", disabled && "is-disabled", selected && "is-selected"]
    .filter(Boolean)
    .join(" ");
  const body = (
    <>
      <span className="card-corner">
        {label && <span className="card-value">{label}</span>}
        <Glyph name={glyphOf(kind)} size={size === "sm" ? 11 : 13} />
      </span>
      <span className="card-center">
        <Glyph name={glyphOf(kind)} size={size === "lg" ? 34 : size === "md" ? 26 : 18} />
      </span>
      {!colored && size !== "sm" && <span className="card-title">{cardName(kind)}</span>}
      {colored && size !== "sm" && <span className="card-value card-value-flip">{label}</span>}
      {badge !== undefined && <span className="card-badge">{badge}</span>}
    </>
  );
  return (
    <div className={`card-slot card-slot-${size}`}>
      <motion.button
        type="button"
        layoutId={layoutId}
        className={classes}
        onClick={onClick}
        disabled={disabled || !onClick}
        aria-label={cardName(kind) + (p !== undefined ? `, ${pct(p)} Prozent` : "")}
        aria-pressed={selected}
        whileHover={onClick && !disabled ? { y: -4 } : undefined}
        whileTap={onClick && !disabled ? { scale: 0.95 } : undefined}
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
      >
        {body}
      </motion.button>
      {p !== undefined && (
        <span className={`gauge ${best ? "gauge-best" : ""}`}>
          <span className="gauge-fill" style={{ transform: `scaleX(${Math.max(p, 0.0)})` }} />
          <span className="gauge-label">{pct(p)}%</span>
        </span>
      )}
    </div>
  );
}

export { pct };
