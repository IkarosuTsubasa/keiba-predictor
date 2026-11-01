import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import { initAnalytics } from "./lib/analytics";
import "./styles.css";

const isSimulator = /^\/keiba\/simulator\/?$/.test(window.location.pathname);
const RaceSimulatorPage = React.lazy(() => import("./components/RaceSimulatorPage"));

if (!isSimulator) initAnalytics();

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    {isSimulator ? (
      <React.Suspense fallback={null}>
        <RaceSimulatorPage />
      </React.Suspense>
    ) : <App />}
  </React.StrictMode>
);
