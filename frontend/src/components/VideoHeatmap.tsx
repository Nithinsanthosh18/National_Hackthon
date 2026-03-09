import React, { useRef, useState, useEffect, useMemo } from 'react';

interface VideoHeatmapProps {
    interactionData: number[]; // Array mapping video seconds to view counts
    duration: number; // Total video duration
    currentTime: number;
    onSeek: (time: number) => void;
}

export const VideoHeatmap: React.FC<VideoHeatmapProps> = ({ interactionData, duration, currentTime, onSeek }) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const [hoverTime, setHoverTime] = useState<number | null>(null);

    // Normalize data to a 0-1 scale for rendering "heat"
    const normalizedData = useMemo(() => {
        if (!interactionData || interactionData.length === 0) return [];
        const maxVal = Math.max(...interactionData, 1); // Avoid division by 0
        return interactionData.map(val => val / maxVal);
    }, [interactionData]);

    const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!containerRef.current || duration <= 0) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        setHoverTime(percentage * duration);
    };

    const handleMouseLeave = () => setHoverTime(null);

    const handleClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (!containerRef.current || duration <= 0) return;
        const rect = containerRef.current.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percentage = Math.max(0, Math.min(1, x / rect.width));
        onSeek(percentage * duration);
    };

    // Ensure we have bars to render even if data is still filling
    const displayBars = normalizedData.length > 0 ? normalizedData : new Array(Math.ceil(duration || 100)).fill(0);

    return (
        <div className="w-full mt-4 space-y-1">
            <div className="flex justify-between text-xs text-muted-foreground mb-1">
                <span>User Engagement Heatmap</span>
                <span>{hoverTime !== null ? `Sec: ${Math.floor(hoverTime)}s` : ''}</span>
            </div>

            <div
                ref={containerRef}
                className="w-full h-12 bg-muted/30 rounded-md flex items-end cursor-pointer relative overflow-hidden group"
                onMouseMove={handleMouseMove}
                onMouseLeave={handleMouseLeave}
                onClick={handleClick}
            >
                {/* Render interactive bars */}
                {displayBars.map((intensity, index) => {
                    // Heatmap color logic: Low=Blue/Cold, High=Red/Hot
                    let colorClass = "bg-primary/20";
                    if (intensity > 0.8) colorClass = "bg-destructive";
                    else if (intensity > 0.5) colorClass = "bg-warning";
                    else if (intensity > 0.2) colorClass = "bg-info";

                    const widthPercent = 100 / displayBars.length;

                    return (
                        <div
                            key={index}
                            className={`${colorClass} transition-all duration-300 hover:opacity-80`}
                            style={{
                                width: `${widthPercent}%`,
                                height: `${Math.max(10, intensity * 100)}%`,
                                borderRight: '1px solid rgba(0,0,0,0.1)'
                            }}
                        />
                    );
                })}

                {/* Current Time Indicator Line */}
                {duration > 0 && (
                    <div
                        className="absolute top-0 bottom-0 w-0.5 bg-foreground z-10 shadow-[0_0_5px_rgba(255,255,255,0.5)] pointer-events-none"
                        style={{ left: `${(currentTime / duration) * 100}%` }}
                    />
                )}
            </div>
        </div>
    );
};
