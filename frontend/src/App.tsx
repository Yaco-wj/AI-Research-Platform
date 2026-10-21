/** 应用外壳：极简顶部与路由。 */

import { Link, Route, Routes } from "react-router-dom";

import { HomePage } from "./pages/HomePage";
import { ResearchPage } from "./pages/ResearchPage";

export default function App() {
  return (
    <div className="app">
      <header className="app__header">
        <Link to="/" className="app__brand">
          AI Research
        </Link>
        <span className="app__tagline">Deep Research</span>
      </header>

      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/research/:taskId" element={<ResearchPage />} />
      </Routes>
    </div>
  );
}
