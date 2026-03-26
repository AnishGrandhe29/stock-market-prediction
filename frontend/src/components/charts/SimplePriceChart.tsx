'use client';

import { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
    createChart, ColorType, IChartApi, ISeriesApi,
    LineStyle, CrosshairMode,
} from 'lightweight-charts';
import { stocksAPI } from '@/lib/api';
import type { PredictionData } from '@/types/dashboard.types';

interface SimplePriceChartProps {
    prediction?: PredictionData;
}

export function SimplePriceChart({ prediction }: SimplePriceChartProps) {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);

    const { data: historyData } = useQuery({
        queryKey: ['price-history'],
        queryFn: () => stocksAPI.getHistory('^NSEI', 60),
    });

    useEffect(() => {
        if (!containerRef.current) return;

        // ── Create chart ──────────────────────────────────────────────────────
        const chart = createChart(containerRef.current, {
            layout: {
                background: { type: ColorType.Solid, color: 'transparent' },
                textColor: '#918f9a',    // --text-muted
            },
            grid: {
                vertLines: { color: 'rgba(70,69,84,0.15)' },
                horzLines: { color: 'rgba(70,69,84,0.15)' },
            },
            width: containerRef.current.clientWidth,
            height: 340,
            rightPriceScale: {
                borderVisible: false,
                textColor: '#918f9a',
            },
            timeScale: {
                borderVisible: false,
                timeVisible: true,
                barSpacing: 10,
            },
            crosshair: {
                mode: CrosshairMode.Normal,
                vertLine: {
                    color: 'rgba(192,193,255,0.4)',
                    labelBackgroundColor: '#c0c1ff',
                    style: LineStyle.Dashed,
                    width: 1,
                },
                horzLine: {
                    color: 'rgba(192,193,255,0.4)',
                    labelBackgroundColor: '#c0c1ff',
                    style: LineStyle.Dashed,
                    width: 1,
                },
            },
            handleScale: true,
            handleScroll: true,
        });

        chartRef.current = chart;

        // ── Candlestick series (emerald/rose) ─────────────────────────────────
        const candleSeries = chart.addCandlestickSeries({
            upColor:       '#4edea3',
            downColor:     '#ffb2b7',
            borderUpColor: '#4edea3',
            borderDownColor: '#ffb2b7',
            wickUpColor:   '#4edea3',
            wickDownColor: '#ffb2b7',
        });

        // ── Volume histogram ───────────────────────────────────────────────────
        const volumeSeries = chart.addHistogramSeries({
            color: '#6366f1',
            priceFormat: { type: 'volume' },
            priceScaleId: '',
        });
        volumeSeries.priceScale().applyOptions({
            scaleMargins: { top: 0.88, bottom: 0 },
        });

        // ── Load price data ────────────────────────────────────────────────────
        if (historyData?.data) {
            const sorted = [...historyData.data].sort((a: any, b: any) =>
                a.date > b.date ? 1 : -1
            );
            const seen = new Set<string>();
            const prices = sorted.filter((p: any) => {
                if (seen.has(p.date)) return false;
                seen.add(p.date);
                return true;
            });

            candleSeries.setData(
                prices.map((p: any) => ({
                    time: p.date,
                    open: p.open, high: p.high, low: p.low, close: p.close,
                }))
            );
            volumeSeries.setData(
                prices.map((p: any) => ({
                    time: p.date,
                    value: p.volume || 0,
                    color: p.close >= p.open
                        ? 'rgba(78,222,163,0.35)'
                        : 'rgba(255,178,183,0.35)',
                }))
            );

            // ── AI Prediction point overlay ────────────────────────────────────
            if (prediction?.predicted_open && prediction?.target_date) {
                const predLineSeries = chart.addLineSeries({
                    color: '#c0c1ff',
                    lineWidth: 2,
                    lineStyle: LineStyle.Dashed,
                    crosshairMarkerVisible: true,
                    crosshairMarkerRadius: 5,
                    crosshairMarkerBorderColor: '#c0c1ff',
                    crosshairMarkerBackgroundColor: '#c0c1ff',
                    lastValueVisible: true,
                    priceLineVisible: false,
                    title: 'AI Pred.',
                });

                // Draw a short dashed projection from last candle to predicted date
                const lastCandle = prices[prices.length - 1];
                if (lastCandle) {
                    predLineSeries.setData([
                        { time: lastCandle.date, value: lastCandle.close },
                        { time: prediction.target_date, value: prediction.predicted_open },
                    ]);
                }
            }

            chart.timeScale().fitContent();
        }

        // ── Resize observer ────────────────────────────────────────────────────
        const ro = new ResizeObserver(() => {
            if (containerRef.current) {
                chart.applyOptions({ width: containerRef.current.clientWidth });
            }
        });
        if (containerRef.current) ro.observe(containerRef.current);

        return () => {
            ro.disconnect();
            chart.remove();
        };
    }, [historyData, prediction]);

    return (
        <div
            ref={containerRef}
            className="chart-container"
            style={{ height: 340 }}
        />
    );
}
