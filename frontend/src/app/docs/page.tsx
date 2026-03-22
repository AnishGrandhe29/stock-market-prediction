'use client';

import {
    BookOpen,
    Brain,
    TrendingUp,
    BarChart3,
    Shield,
    Zap,
    HelpCircle,
    Code2,
    ChevronDown,
    ChevronRight,
    LineChart,
    Eye,
    Star,
    Bell,
    FileText,
} from 'lucide-react';
import { useState } from 'react';

interface AccordionItemProps {
    question: string;
    answer: string;
}

function AccordionItem({ question, answer }: AccordionItemProps) {
    const [open, setOpen] = useState(false);
    return (
        <div className="border border-surface-200 dark:border-surface-700 rounded-xl overflow-hidden">
            <button
                onClick={() => setOpen(!open)}
                className="w-full flex items-center justify-between px-5 py-4 text-left hover:bg-surface-50 dark:hover:bg-surface-700/50 transition-colors"
            >
                <span className="font-medium text-surface-900 dark:text-white text-sm">{question}</span>
                {open
                    ? <ChevronDown className="w-4 h-4 text-primary-500 flex-shrink-0" />
                    : <ChevronRight className="w-4 h-4 text-surface-400 flex-shrink-0" />}
            </button>
            {open && (
                <div className="px-5 pb-4 text-sm text-surface-600 dark:text-surface-400 leading-relaxed border-t border-surface-100 dark:border-surface-700 pt-3">
                    {answer}
                </div>
            )}
        </div>
    );
}

const sections = [
    {
        id: 'getting-started',
        icon: BookOpen,
        color: 'text-primary-500',
        bg: 'bg-primary-100 dark:bg-primary-900/30',
        title: 'Getting Started',
        content: (
            <div className="space-y-4 text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                <p>
                    Welcome to the <span className="font-semibold text-surface-900 dark:text-white">NIFTY 50 Index Predictor</span> — an AI-powered platform
                    that uses Multimodal Temporal Convolutional Networks (TCN) with Adaptive Fusion to forecast NIFTY 50 index movements.
                </p>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mt-4">
                    {[
                        { icon: BarChart3, label: 'Dashboard', desc: 'Real-time index overview with latest AI prediction' },
                        { icon: LineChart, label: 'Live Chart', desc: 'Candlestick chart with technical indicator overlays' },
                        { icon: Brain, label: 'Predictions', desc: 'Daily forecasts with confidence intervals and signals' },
                        { icon: Eye, label: 'XAI Insights', desc: 'Explainability panel showing which features drove predictions' },
                        { icon: Star, label: 'Watchlist', desc: 'Track symbols and set custom price targets' },
                        { icon: Bell, label: 'Alerts', desc: 'Configure notification thresholds for price and prediction events' },
                    ].map(({ icon: Icon, label, desc }) => (
                        <div key={label} className="flex items-start gap-3 p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                            <Icon className="w-4 h-4 text-primary-500 mt-0.5 flex-shrink-0" />
                            <div>
                                <p className="font-medium text-surface-900 dark:text-white text-xs">{label}</p>
                                <p className="text-xs text-surface-500">{desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        ),
    },
    {
        id: 'model',
        icon: Brain,
        color: 'text-purple-500',
        bg: 'bg-purple-100 dark:bg-purple-900/30',
        title: 'About the AI Model',
        content: (
            <div className="space-y-4 text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                <p>
                    The prediction engine is <span className="font-semibold text-surface-900 dark:text-white">NIFTY50-Multimodal-TCN</span>, a
                    custom deep learning model with ~847K parameters trained on NIFTY 50 data from 2010–2025.
                </p>
                <div className="p-4 bg-purple-50 dark:bg-purple-900/20 rounded-xl border border-purple-200 dark:border-purple-800">
                    <h4 className="font-semibold text-purple-800 dark:text-purple-300 mb-2 text-xs uppercase tracking-wide">Architecture at a Glance</h4>
                    <div className="space-y-2">
                        {[
                            ['Price Encoder', '6-layer Dilated TCN — 60-day OHLCV sequences → 128-dim embedding'],
                            ['Sentiment Encoder', '2-layer MLP — news + Reddit sentiment scores → 128-dim embedding'],
                            ['Technical Encoder', '3-layer MLP — RSI, MACD, ADX, ATR, Stochastic → 128-dim embedding'],
                            ['Adaptive Fusion Gate', 'Dynamic modality weighting with learned temperature softmax'],
                            ['Prediction Head', 'Point return % + 3 quantiles (5/50/95th) + direction probabilities'],
                        ].map(([name, desc]) => (
                            <div key={name} className="flex gap-2">
                                <span className="flex-shrink-0 text-purple-500 font-bold">›</span>
                                <span><span className="font-medium text-surface-900 dark:text-white">{name}:</span> {desc}</span>
                            </div>
                        ))}
                    </div>
                </div>
                <p>
                    Predictions are clamped to <code className="bg-surface-100 dark:bg-surface-700 px-1.5 py-0.5 rounded text-xs">±2%</code> daily
                    return to prevent unrealistic forecasts. The confidence score blends direction probability (60%) with uncertainty (40%).
                </p>
            </div>
        ),
    },
    {
        id: 'predictions',
        icon: TrendingUp,
        color: 'text-emerald-500',
        bg: 'bg-emerald-100 dark:bg-emerald-900/30',
        title: 'Reading Predictions',
        content: (
            <div className="space-y-4 text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                <p>Each prediction includes three key outputs:</p>
                <div className="space-y-3">
                    {[
                        {
                            label: 'Point Prediction',
                            color: 'border-blue-400',
                            bg: 'bg-blue-50 dark:bg-blue-900/20',
                            desc: 'The model\'s best estimate of the next trading day\'s return as a percentage (e.g., +0.42% or -0.18%).',
                        },
                        {
                            label: 'Confidence Interval (Quantiles)',
                            color: 'border-amber-400',
                            bg: 'bg-amber-50 dark:bg-amber-900/20',
                            desc: 'The 5th–95th percentile range shows a 90% probability band. A tighter band means the model is more certain.',
                        },
                        {
                            label: 'Trading Signal',
                            color: 'border-emerald-400',
                            bg: 'bg-emerald-50 dark:bg-emerald-900/20',
                            desc: 'BUY (predicted return > +0.3%), HOLD (−0.3% to +0.3%), or SELL (< −0.3%) based on direction probability.',
                        },
                    ].map(({ label, color, bg, desc }) => (
                        <div key={label} className={`p-4 rounded-xl border-l-4 ${color} ${bg}`}>
                            <p className="font-semibold text-surface-900 dark:text-white mb-1">{label}</p>
                            <p>{desc}</p>
                        </div>
                    ))}
                </div>
                <p className="text-xs text-surface-400 italic">
                    ⚠ Predictions are for informational purposes only and do not constitute financial advice.
                </p>
            </div>
        ),
    },
    {
        id: 'xai',
        icon: Eye,
        color: 'text-amber-500',
        bg: 'bg-amber-100 dark:bg-amber-900/30',
        title: 'Explainable AI (XAI)',
        content: (
            <div className="space-y-4 text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                <p>
                    The XAI Insights page reveals <span className="font-semibold text-surface-900 dark:text-white">why</span> the model made a particular prediction using four complementary methods:
                </p>
                <div className="space-y-2">
                    {[
                        ['Adaptive Fusion Weights', 'Real-time breakdown of how much each modality (price / sentiment / technical) contributed to today\'s prediction.'],
                        ['SHAP Feature Importance', 'Perturbation-based attribution showing which individual features pushed the prediction higher or lower.'],
                        ['Gradient Attribution', 'Gradient-based sensitivity showing which time steps in the 60-day window were most influential.'],
                        ['Natural Language Summary', 'A plain-English explanation generated automatically from the model outputs and feature attributions.'],
                    ].map(([name, desc]) => (
                        <div key={name} className="flex items-start gap-3 p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg">
                            <Zap className="w-4 h-4 text-amber-500 mt-0.5 flex-shrink-0" />
                            <div>
                                <p className="font-medium text-surface-900 dark:text-white">{name}</p>
                                <p className="text-xs mt-0.5">{desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        ),
    },
    {
        id: 'api',
        icon: Code2,
        color: 'text-cyan-500',
        bg: 'bg-cyan-100 dark:bg-cyan-900/30',
        title: 'API Reference',
        content: (
            <div className="space-y-4 text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                <p>The backend exposes a RESTful API at <code className="bg-surface-100 dark:bg-surface-700 px-2 py-0.5 rounded text-xs">/api/v1</code>. Key endpoints:</p>
                <div className="space-y-2">
                    {[
                        { method: 'GET', path: '/api/v1/predictions/latest', desc: 'Latest NIFTY 50 prediction with confidence and signal' },
                        { method: 'GET', path: '/api/v1/predictions/history', desc: '30-day prediction history with accuracy metrics' },
                        { method: 'GET', path: '/api/v1/stocks/nifty50', desc: 'Current NIFTY 50 index price and change' },
                        { method: 'GET', path: '/api/v1/model/info', desc: 'Model architecture metadata and version' },
                        { method: 'GET', path: '/api/v1/news/sentiment', desc: 'Latest market news with sentiment scores' },
                        { method: 'WS', path: '/api/v1/ws/live', desc: 'WebSocket stream for real-time price updates' },
                    ].map(({ method, path, desc }) => (
                        <div key={path} className="flex items-start gap-3 p-3 bg-surface-50 dark:bg-surface-700/50 rounded-lg font-mono">
                            <span className={`flex-shrink-0 text-xs font-bold px-2 py-0.5 rounded ${method === 'GET' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300' : 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300'}`}>
                                {method}
                            </span>
                            <div>
                                <p className="text-xs text-surface-900 dark:text-white">{path}</p>
                                <p className="text-xs text-surface-500 font-sans mt-0.5">{desc}</p>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        ),
    },
    {
        id: 'disclaimer',
        icon: Shield,
        color: 'text-red-500',
        bg: 'bg-red-100 dark:bg-red-900/30',
        title: 'Disclaimer & Risk Warning',
        content: (
            <div className="space-y-3 text-sm text-surface-600 dark:text-surface-400 leading-relaxed">
                <div className="p-4 bg-red-50 dark:bg-red-900/20 rounded-xl border border-red-200 dark:border-red-800">
                    <p className="font-semibold text-red-700 dark:text-red-400 mb-2">Important Notice</p>
                    <ul className="space-y-1.5 list-disc list-inside text-xs">
                        <li>This platform is a <strong>research / educational tool</strong> and does NOT constitute financial advice.</li>
                        <li>AI predictions are probabilistic and can be wrong. Past performance is not indicative of future results.</li>
                        <li>Stock markets are inherently unpredictable. Never invest money you cannot afford to lose.</li>
                        <li>Always consult a SEBI-registered financial advisor before making investment decisions.</li>
                    </ul>
                </div>
                <p>
                    This system was built as an academic final-year project demonstrating the application of Multimodal Deep Learning
                    and Explainable AI to financial time series forecasting.
                </p>
            </div>
        ),
    },
];

const faqs: AccordionItemProps[] = [
    {
        question: 'How often is the prediction updated?',
        answer: 'Predictions are generated once per trading day after market close (after 3:30 PM IST). During market hours a cached result is served and refreshed every 5 minutes.',
    },
    {
        question: 'What does "Confidence" mean?',
        answer: 'Confidence is a blended score: 60% weight on the model\'s direction probability (how sure it is of BUY/HOLD/SELL) and 40% weight on the inverse of prediction uncertainty (narrower quantile interval = higher confidence).',
    },
    {
        question: 'What data does the model use?',
        answer: 'Three modalities: (1) 60 days of OHLCV price history, (2) aggregated news and Reddit sentiment scores, and (3) technical indicators — RSI, MACD, Stochastic %K, ADX, and ATR.',
    },
    {
        question: 'Why is the prediction clamped to ±2%?',
        answer: 'NIFTY 50 rarely moves more than ±2% in a single day without circuit breaker events. Clamping prevents the model from generating unrealistic tail forecasts that would mislead users.',
    },
    {
        question: 'Can I use this platform for live trading?',
        answer: 'No. This is a research project. Predictions are directional estimates, not trading signals. Do not make financial decisions based solely on this tool.',
    },
    {
        question: 'Why TCN instead of LSTM?',
        answer: 'Temporal Convolutional Networks (TCN) can be trained in parallel (unlike sequential LSTM) and achieve equivalent or better performance on financial time series, with faster inference.',
    },
];

export default function DocumentationPage() {
    const [activeSection, setActiveSection] = useState('getting-started');

    return (
        <div className="space-y-6">
            {/* Header */}
            <div className="flex items-center gap-3">
                <div className="p-3 bg-primary-100 dark:bg-primary-900/30 rounded-xl">
                    <BookOpen className="w-6 h-6 text-primary-500" />
                </div>
                <div>
                    <h1 className="text-3xl font-bold text-surface-900 dark:text-white">Documentation</h1>
                    <p className="text-surface-500 mt-0.5">Learn how to use the NIFTY 50 Index Predictor platform</p>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Sidebar Nav */}
                <div className="lg:col-span-1">
                    <div className="card p-4 sticky top-24">
                        <p className="text-xs font-semibold uppercase tracking-widest text-surface-400 mb-3">Sections</p>
                        <nav className="space-y-1">
                            {sections.map((s) => {
                                const Icon = s.icon;
                                const isActive = activeSection === s.id;
                                return (
                                    <button
                                        key={s.id}
                                        onClick={() => {
                                            setActiveSection(s.id);
                                            document.getElementById(s.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                                        }}
                                        className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-left text-sm transition-all ${isActive
                                            ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 font-medium'
                                            : 'text-surface-600 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-surface-700'
                                            }`}
                                    >
                                        <Icon className={`w-4 h-4 flex-shrink-0 ${isActive ? 'text-primary-500' : ''}`} />
                                        {s.title}
                                    </button>
                                );
                            })}
                            <div className="border-t border-surface-200 dark:border-surface-700 my-2" />
                            <button
                                onClick={() => document.getElementById('faq')?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
                                className="w-full flex items-center gap-2.5 px-3 py-2.5 rounded-lg text-left text-sm text-surface-600 dark:text-surface-400 hover:bg-surface-100 dark:hover:bg-surface-700 transition-all"
                            >
                                <HelpCircle className="w-4 h-4 flex-shrink-0" />
                                FAQ
                            </button>
                        </nav>
                    </div>
                </div>

                {/* Main Content */}
                <div className="lg:col-span-3 space-y-6">
                    {sections.map((s) => {
                        const Icon = s.icon;
                        return (
                            <div key={s.id} id={s.id} className="card p-6 scroll-mt-24">
                                <div className="flex items-center gap-3 mb-5">
                                    <div className={`p-2.5 rounded-xl ${s.bg}`}>
                                        <Icon className={`w-5 h-5 ${s.color}`} />
                                    </div>
                                    <h2 className="text-xl font-semibold text-surface-900 dark:text-white">{s.title}</h2>
                                </div>
                                {s.content}
                            </div>
                        );
                    })}

                    {/* FAQ Section */}
                    <div id="faq" className="card p-6 scroll-mt-24">
                        <div className="flex items-center gap-3 mb-5">
                            <div className="p-2.5 rounded-xl bg-indigo-100 dark:bg-indigo-900/30">
                                <HelpCircle className="w-5 h-5 text-indigo-500" />
                            </div>
                            <h2 className="text-xl font-semibold text-surface-900 dark:text-white">Frequently Asked Questions</h2>
                        </div>
                        <div className="space-y-3">
                            {faqs.map((faq) => (
                                <AccordionItem key={faq.question} question={faq.question} answer={faq.answer} />
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
