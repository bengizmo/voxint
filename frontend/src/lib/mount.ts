// The data-props contract: every island mount point carries its server-rendered
// props as a JSON string in `data-props`. Parse once at mount.
export function readProps<T>(el: HTMLElement): T {
  const raw = el.dataset.props;
  if (!raw) throw new Error("island mount point missing data-props");
  return JSON.parse(raw) as T;
}
