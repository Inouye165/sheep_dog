import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import { StudioApp } from "./studio/StudioApp";
import "./styles/global.css";

function isStudioRoute(): boolean {
  if (typeof window === "undefined") return false;
  const path = window.location.pathname.toLowerCase();
  const params = new URLSearchParams(window.location.search);
  const hash = window.location.hash.toLowerCase();

  if (
    path === "/v2" ||
    path.startsWith("/v2/") ||
    path.endsWith("/v2.html") ||
    path === "/studio" ||
    path.startsWith("/studio/")
  ) {
    return true;
  }
  const uiParam = (params.get("ui") ?? params.get("view") ?? "").toLowerCase();
  if (uiParam === "v2" || uiParam === "studio") {
    return true;
  }
  if (hash === "#v2" || hash === "#studio") {
    return true;
  }
  return false;
}

export function RootRouter() {
  const [useStudio, setUseStudio] = useState<boolean>(() => isStudioRoute());

  useEffect(() => {
    const onUrlChange = () => setUseStudio(isStudioRoute());
    window.addEventListener("popstate", onUrlChange);
    window.addEventListener("hashchange", onUrlChange);
    return () => {
      window.removeEventListener("popstate", onUrlChange);
      window.removeEventListener("hashchange", onUrlChange);
    };
  }, []);

  return useStudio ? <StudioApp /> : <App />;
}

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <RootRouter />
  </React.StrictMode>
);
