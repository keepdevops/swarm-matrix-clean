import { cpp } from "@codemirror/lang-cpp";
import { javascript } from "@codemirror/lang-javascript";
import { python } from "@codemirror/lang-python";
import { Compartment, EditorState } from "@codemirror/state";
import { oneDark } from "@codemirror/theme-one-dark";
import { EditorView, keymap, lineNumbers, highlightActiveLine } from "@codemirror/view";
import { defaultKeymap, history, historyKeymap } from "@codemirror/commands";
import { useEffect, useRef } from "react";

export type Language = "cpp" | "python" | "javascript" | "plaintext";

type Props = {
  value: string;
  onChange: (next: string) => void;
  language: Language;
};

const languageCompartment = new Compartment();

function langExt(lang: Language) {
  switch (lang) {
    case "cpp": return cpp();
    case "python": return python();
    case "javascript": return javascript({ typescript: true });
    case "plaintext": default: return [];
  }
}

export function Editor({ value, onChange, language }: Props) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const viewRef = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  onChangeRef.current = onChange;

  // Mount once. CodeMirror manages its own state; we sync on prop changes below.
  useEffect(() => {
    if (!hostRef.current) return;
    const state = EditorState.create({
      doc: value,
      extensions: [
        lineNumbers(),
        highlightActiveLine(),
        history(),
        keymap.of([...defaultKeymap, ...historyKeymap]),
        oneDark,
        languageCompartment.of(langExt(language)),
        EditorView.updateListener.of((u) => {
          if (u.docChanged) onChangeRef.current(u.state.doc.toString());
        }),
        EditorView.theme({
          "&": { fontSize: "13px" },
          ".cm-scroller": { fontFamily: "var(--font-mono)" },
        }),
      ],
    });
    viewRef.current = new EditorView({ state, parent: hostRef.current });
    return () => {
      viewRef.current?.destroy();
      viewRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Reconfigure language without rebuilding the view.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    view.dispatch({
      effects: languageCompartment.reconfigure(langExt(language)),
    });
  }, [language]);

  // Push external value changes (e.g., agent template) into the doc.
  useEffect(() => {
    const view = viewRef.current;
    if (!view) return;
    const current = view.state.doc.toString();
    if (current === value) return;
    view.dispatch({
      changes: { from: 0, to: current.length, insert: value },
    });
  }, [value]);

  return <div className="editor-host" ref={hostRef} />;
}
