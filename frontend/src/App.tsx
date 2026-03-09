import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { MonitoringProvider } from "@/context/MonitoringContext";
import Index from "./pages/Index";
import CamerasPage from "./pages/CamerasPage";
import AlertsPage from "./pages/AlertsPage";
import HeatmapPage from "./pages/HeatmapPage";
import LogsPage from "./pages/LogsPage";
import DetectionRulesPage from "./pages/DetectionRulesPage";
import WatchlistPage from "./pages/WatchlistPage";
import ReportsPage from "./pages/ReportsPage";
import ArchivePage from "./pages/ArchivePage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <MonitoringProvider>
          <Routes>
            <Route path="/" element={<Index />} />
            <Route path="/cameras" element={<CamerasPage />} />
            <Route path="/alerts" element={<AlertsPage />} />
            <Route path="/heatmap" element={<HeatmapPage />} />
            <Route path="/logs" element={<LogsPage />} />
            <Route path="/rules" element={<DetectionRulesPage />} />
            <Route path="/watchlist" element={<WatchlistPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/archive" element={<ArchivePage />} />
            <Route path="*" element={<NotFound />} />
          </Routes>
        </MonitoringProvider>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
