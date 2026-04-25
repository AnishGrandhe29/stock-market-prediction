'use client';

import { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, LineStyle, CrosshairMode } from 'lightweight-charts';
import type { PredictionData } from '@/types/dashboard.types';

interface SimplePriceChartProps {
    prediction?: PredictionData;
}

// Calculate Simple Moving Average
function calculateSMA(data: number[], period: number): (number | null)[] {
    const sma: (number | null)[] = [];
    for (let i = 0; i < data.length; i++) {
        if (i < period - 1) {
            sma.push(null);
        } else {
            const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
            sma.push(sum / period);
        }
    }
    return sma;
}

export function SimplePriceChart({ prediction }: SimplePriceChartProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    useEffect(() => {
        if (!containerRef.current) return;

        const isDark = document.documentElement.classList.contains('dark');
        const textColor = isDark ? '#918f9a' : '#6a6a70';
        const gridColor = isDark ? 'rgba(70,69,84,0.1)' : 'rgba(0,0,0,0.05)';

        // ── Create chart ──────────────────────────────────────────────────────
        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: textColor,
            },
            grid: {
                vertLines: { color: gridColor },
                horzLines: { color: gridColor },
            },
            width: containerRef.current.clientWidth,
            height: 340,
            rightPriceScale: {
                borderVisible: false,
                scaleMargins: {
                    top: 0.1,
                    bottom: 0.2, // Leave space for volume
                },
            },
            timeScale: {
                borderVisible: false,
                timeVisible: true,
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    labelBackgroundColor: '#6366f1',
                    style: LineStyle.Dashed,
                    width: 1,
                },
                horzLine: {
                    labelBackgroundColor: '#6366f1',
                    style: LineStyle.Dashed,
                    width: 1,
                },
            },
            handleScale: true,
            handleScroll: true,
        });

        chartRef.current = chart;

        // ── Candlestick series ────────────────────────────────────────────────
        const candleSeries = chart.addCandlestickSeries({
            upColor:       '#10b981',
            downColor:     '#ef4444',
            borderUpColor: '#10b981',
            borderDownColor: '#ef4444',
            wickUpColor:   '#10b981',
            wickDownColor: '#ef4444',
        });

        // ── MA 30 series ──────────────────────────────────────────────────────
        const ma30Series = chart.addLineSeries({
            color: '#f59e0b',
            lineWidth: 1.5,
            title: 'MA 30',
            lastValueVisible: false,
            priceLineVisible: false,
        });

        // ── Volume histogram ───────────────────────────────────────────────────
        const volumeSeries = chart.addHistogramSeries({
            color: '#6366f1',
            priceFormat: { type: 'volume' },
            priceScaleId: '',
        });
        volumeSeries.priceScale().applyOptions({
            scaleMargins: { top: 0.85, bottom: 0 },
        });

        // ── Prediction series ─────────────────────────────────────────────────
        const predictionSeries = chart.addLineSeries({
            color: '#6366f1',
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
            lastValueVisible: true,
            priceLineVisible: false,
            title: 'AI Predicted Open',
        });

        // ── Load data ─────────────────────────────────────────────────────────
        if (prediction?.input_features?.historical_data) {
            const rawData = [...prediction.input_features.historical_data].reverse();
            
            const candleData = rawData.map((d: any) => ({
                time: d.date,
                open: d.open,
                high: d.high,
                low: d.low,
                close: d.close,
            }));

            const volumeData = rawData.map((d: any) => ({
                time: d.date,
                value: d.volume || 0,
                color: d.close >= d.open ? 'rgba(16, 185, 129, 0.4)' : 'rgba(239, 68, 68, 0.4)',
            }));

            const closePrices = rawData.map((d: any) => d.close);
            const ma30Values = calculateSMA(closePrices, 30);
            const ma30Data = rawData.map((d: any, i: number) => ({
                time: d.date,
                value: ma30Values[i],
            })).filter((d: any) => d.value !== null);

            candleSeries.setData(candleData);
            volumeSeries.setData(volumeData);
            ma30Series.setData(ma30Data);

            if (prediction.predicted_open) {
                const lastCandle = candleData[candleData.length - 1];
                
                // Estimate next trading date
                const lastDate = new Date(lastCandle.time as string);
                const nextDate = new Date(lastDate);
                nextDate.setDate(nextDate.getDate() + (lastDate.getDay() === 5 ? 3 : 1));
                const nextDateStr = nextDate.toISOString().split('T')[0];

                predictionSeries.setData([
                    { time: lastCandle.time, value: lastCandle.close },
                    { time: nextDateStr, value: prediction.predicted_open }
                ]);
            }

            chart.timeScale().fitContent();
        }

        // Resize handler
        const handleResize = () => {
            if (containerRef.current) {
                chart.applyOptions({ width: containerRef.current.clientWidth });
            }
        };
        window.addEventListener('resize', handleResize);

        // Theme observer
        const observer = new MutationObserver(() => {
            const dark = document.documentElement.classList.contains('dark');
            chart.applyOptions({
                layout: {
                    textColor: dark ? '#918f9a' : '#6a6a70',
                },
                grid: {
                    vertLines: { color: dark ? 'rgba(70,69,84,0.1)' : 'rgba(0,0,0,0.05)' },
                    horzLines: { color: dark ? 'rgba(70,69,84,0.1)' : 'rgba(0,0,0,0.05)' },
                },
            });
        });
        observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });

        return () => {
            window.removeEventListener('resize', handleResize);
            observer.disconnect();
            chart.remove();
        };
    }, [prediction]);

    return (
        <div ref={containerRef} className="w-full relative" />
    );
}
