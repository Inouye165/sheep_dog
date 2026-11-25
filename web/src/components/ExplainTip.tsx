import { useEffect, useRef, useState } from "react";

export interface ExplainTipProps {
  title: string;
  plain: string;
  technical?: string[];
  label?: string;
  buttonText?: string;
  align?: "left" | "right";
}

export function ExplainTip({
  title,
  plain,
  technical,
  label,
  buttonText,
  align = "left",
}: ExplainTipProps) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLSpanElement | null>(null);

  useEffect(() => {
    if (!open) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    const handleMouseDown = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    document.addEventListener("mousedown", handleMouseDown);
    return () => {
      document.removeEventListener("keydown", handleKeyDown);
      document.removeEventListener("mousedown", handleMouseDown);
    };
  }, [open]);

  return (
    <span className="ui-explain" ref={containerRef}>
      <button
        type="button"
        className={`ui-explain__trigger${buttonText ? " ui-explain__trigger--pill" : ""}${
          open ? " ui-explain__trigger--open" : ""
        }`}
        aria-expanded={open}
        aria-label={label ?? `Explain ${title}`}
        onClick={() => setOpen((prev) => !prev)}
      >
        {buttonText ? (
          <>
            <span className="ui-explain__badge" aria-hidden="true">
              ?
            </span>
            <span>{buttonText}</span>
          </>
        ) : (
          "?"
        )}
      </button>

      {open ? (
        <div
          className={`ui-explain__popover ui-explain__popover--${align}`}
          role="dialog"
          aria-label={title}
        >
          <div className="ui-explain__header">
            <strong className="ui-explain__title">{title}</strong>
            <button
              type="button"
              className="ui-explain__close"
              aria-label="Close explanation"
              onClick={() => setOpen(false)}
            >
              ×
            </button>
          </div>
          <div className="ui-explain__section">
            <span className="ui-explain__section-label">Plain English</span>
            <p className="ui-explain__plain">{plain}</p>
          </div>
          {technical && technical.length > 0 ? (
            <div className="ui-explain__section">
              <span className="ui-explain__section-label">For Data Scientists</span>
              <ul className="ui-explain__list">
                {technical.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      ) : null}
    </span>
  );
}

export interface ProgressPageGuideModalProps {
  isOpen: boolean;
  onClose: () => void;
  stage: number;
  promotionTargetPct: number;
}

const TAB_MAP = [
  {
    name: "Learning Curve",
    description:
      "Tracks how often the 3 dogs pass the 10-seed test, how fast they finish, and their episode return over time.",
  },
  {
    name: "Checkpoints",
    description:
      "Lists every saved model snapshot with pass rate, average sheep penned, steps, timeouts, and replay links.",
  },
  {
    name: "Seeds & Outcomes",
    description:
      "Shows active training/evaluation time and per-seed pass/fail tables across training and formal evaluations.",
  },
  {
    name: "Speed & Timing",
    description:
      "Monitors learning throughput (steps/s), evaluation overhead per checkpoint, and cross-run stage promotion times.",
  },
  {
    name: "Milestones & Video",
    description:
      "Lets you browse milestone replays for completed stages and export video bundles.",
  },
];

const GLOSSARY = [
  {
    term: "Curriculum Stage",
    definition:
      "A difficulty level in training. Dogs graduate to harder scenarios only after proving consistent reliability.",
  },
  {
    term: "Formal 10-Seed Eval",
    definition:
      "A fixed suite of 10 test scenarios run periodically without exploration noise to measure true policy skill.",
  },
  {
    term: "Promotion Gate",
    definition:
      "The automatic graduation rule requiring both enough recent evaluations and consecutive qualifying scores.",
  },
  {
    term: "Checkpoint",
    definition:
      "A saved snapshot of the neural network weights at a specific training episode, paired with its evaluation metrics.",
  },
  {
    term: "World Steps vs Policy Steps",
    definition:
      "World steps count simulation ticks in one episode (fewer = faster herding); policy steps count total RL transitions learned across workers.",
  },
  {
    term: "Episode Return",
    definition:
      "Sum of shaped and completion rewards earned in an episode. Useful for spotting reward trends, though pass rate is the true success criterion.",
  },
];

export function ProgressPageGuideModal({
  isOpen,
  onClose,
  stage,
  promotionTargetPct,
}: ProgressPageGuideModalProps) {
  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <div
      className="ui-guide-modal__backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <div
        className="ui-guide-modal"
        role="dialog"
        aria-modal="true"
        aria-label="How to read the Progress page"
      >
        <div className="ui-guide-modal__header">
          <div>
            <h2 className="ui-guide-modal__title">How to read the Progress page</h2>
            <p className="ui-guide-modal__summary">
              This page tracks how reliably the 3 dogs pen all 6 sheep on Stage {stage} and when the
              model earns promotion (clearing the {promotionTargetPct}% bar).
            </p>
          </div>
          <button
            type="button"
            className="ui-guide-modal__close"
            aria-label="Close page guide"
            onClick={onClose}
          >
            ×
          </button>
        </div>

        <div className="ui-guide-modal__body">
          <section className="ui-guide-modal__section">
            <h3 className="ui-guide-modal__section-title">What lives in each tab</h3>
            <div className="ui-guide-modal__tab-grid">
              {TAB_MAP.map((tab) => (
                <div key={tab.name} className="ui-guide-modal__tab-card">
                  <strong>{tab.name}</strong>
                  <p>{tab.description}</p>
                </div>
              ))}
            </div>
          </section>

          <section className="ui-guide-modal__section">
            <h3 className="ui-guide-modal__section-title">Key terms</h3>
            <dl className="ui-guide-modal__glossary">
              {GLOSSARY.map((item) => (
                <div key={item.term} className="ui-guide-modal__glossary-item">
                  <dt>{item.term}</dt>
                  <dd>{item.definition}</dd>
                </div>
              ))}
            </dl>
          </section>
        </div>
      </div>
    </div>
  );
}
