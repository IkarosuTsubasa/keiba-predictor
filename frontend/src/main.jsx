import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initAnalytics } from "./lib/analytics";
import "./styles.css";

const path = window.location.pathname.replace(/\/+$/, "");
const isSimulator = /^\/keiba\/simulator(?:\/embed)?$/.test(path) || /^\/keiba\/open\/simulator\/\d{12}$/.test(path);
const isLab = import.meta.env.DEV && new URLSearchParams(window.location.search).get("lab") === "1";
const SimulatorProductionPage = React.lazy(() => import("./components/SimulatorProductionPage"));
const RaceSimulatorPage = React.lazy(() => import("./components/RaceSimulatorPage"));

if (!path.endsWith("/embed") && !isLab) initAnalytics();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {isSimulator ? (
      <React.Suspense fallback={null}>
        {isLab ? <RaceSimulatorPage /> : <SimulatorProductionPage embedded={path.endsWith("/embed")} landing={path.includes("/open/simulator/")} />}
      </React.Suspense>
    ) : <App />}
  </React.StrictMode>
);
