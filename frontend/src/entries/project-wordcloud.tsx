import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { WordCloud, type TermDatum } from "../components/WordCloud";
import { readProps } from "../lib/mount";

interface ProjectWordCloudProps {
  terms: TermDatum[];
  projectId: string;
}

export function mount(el: HTMLElement): void {
  const props = readProps<ProjectWordCloudProps>(el);
  const handleClick = (term: string) => {
    window.location.href = `/explore?q=${encodeURIComponent(term)}&project=${encodeURIComponent(props.projectId)}`;
  };
  createRoot(el).render(
    <StrictMode>
      <WordCloud terms={props.terms} onTermClick={handleClick} />
    </StrictMode>,
  );
}
