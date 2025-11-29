#pragma once

#ifdef ARGUS_ONNX_ENABLED

#include <onnxruntime_cxx_api.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>
#include <array>
#include <memory>
#include <fstream>
#include <stdexcept>
#include <cmath>
#include <deque>
#include <unordered_map>

namespace argus::ml {

/**
 * Feature normalization statistics loaded from training.
 */
struct NormStats {
    double mean = 0.0;
    double std = 1.0;
};

/**
 * Prediction result from the model.
 */
struct Prediction {
    double predicted_return;      // Predicted return over horizon
    double confidence;            // Model confidence (0-1)
    int direction;                // 1 = bullish, -1 = bearish, 0 = neutral
    std::string signal;           // "BUY", "SELL", "HOLD"
};

/**
 * OHLCV bar for feature calculation.
 */
struct Bar {
    double open;
    double high;
    double low;
    double close;
    double volume;
    int64_t timestamp;
};

/**
 * OnnxPredictor - Runs ML model inference using ONNX Runtime.
 *
 * This class:
 * 1. Loads the trained ONNX model
 * 2. Maintains a sliding window of historical bars
 * 3. Computes technical indicators (features)
 * 4. Normalizes features using training statistics
 * 5. Runs inference to predict future returns
 */
class OnnxPredictor {
public:
    OnnxPredictor() = default;

    /**
     * Initialize the predictor with model and config paths.
     * @param model_path Path to the ONNX model file
     * @param config_path Path to the training config JSON
     * @param norm_stats_path Path to normalization statistics JSON
     */
    bool initialize(const std::string& model_path,
                   const std::string& config_path,
                   const std::string& norm_stats_path) {
        try {
            // Load config
            if (!load_config(config_path)) {
                spdlog::error("Failed to load config from {}", config_path);
                return false;
            }

            // Load normalization stats
            if (!load_norm_stats(norm_stats_path)) {
                spdlog::error("Failed to load norm stats from {}", norm_stats_path);
                return false;
            }

            // Initialize ONNX Runtime
            env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ArgusPredictor");

            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef ARGUS_ONNX_CUDA
            // Try to use CUDA if available
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0;
                session_options.AppendExecutionProvider_CUDA(cuda_options);
                spdlog::info("ONNX: Using CUDA execution provider");
            } catch (const Ort::Exception& e) {
                spdlog::warn("ONNX: CUDA not available, falling back to CPU: {}", e.what());
            }
#endif

            // Load the model
            std::wstring wmodel_path(model_path.begin(), model_path.end());
            session_ = std::make_unique<Ort::Session>(*env_, wmodel_path.c_str(), session_options);

            // Get input/output info
            Ort::AllocatorWithDefaultOptions allocator;

            auto input_name = session_->GetInputNameAllocated(0, allocator);
            input_name_ = input_name.get();

            auto output_name = session_->GetOutputNameAllocated(0, allocator);
            output_name_ = output_name.get();

            auto input_shape = session_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (input_shape.size() >= 2) {
                sequence_length_ = static_cast<size_t>(input_shape[1]);
                num_features_ = static_cast<size_t>(input_shape[2]);
            }

            spdlog::info("ONNX model loaded: {} -> {}", input_name_, output_name_);
            spdlog::info("Input shape: [batch, {}, {}]", sequence_length_, num_features_);

            initialized_ = true;
            return true;

        } catch (const Ort::Exception& e) {
            spdlog::error("ONNX initialization failed: {}", e.what());
            return false;
        } catch (const std::exception& e) {
            spdlog::error("Initialization failed: {}", e.what());
            return false;
        }
    }

    /**
     * Add a new bar to the history buffer.
     * Call this as new market data arrives.
     */
    void add_bar(const Bar& bar) {
        bar_history_.push_back(bar);

        // Keep only sequence_length + extra for indicator calculation
        size_t max_history = sequence_length_ + 50;
        while (bar_history_.size() > max_history) {
            bar_history_.pop_front();
        }
    }

    /**
     * Check if we have enough data to make a prediction.
     */
    bool can_predict() const {
        return initialized_ && bar_history_.size() >= sequence_length_ + 20;
    }

    /**
     * Run prediction on the current bar history.
     * @return Prediction result with predicted return and signal
     */
    Prediction predict() {
        if (!can_predict()) {
            return {0.0, 0.0, 0, "HOLD"};
        }

        try {
            // Compute features for the sequence
            auto features = compute_features();

            // Normalize features
            normalize(features);

            // Prepare input tensor
            std::vector<int64_t> input_shape = {1, static_cast<int64_t>(sequence_length_),
                                                 static_cast<int64_t>(num_features_)};

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, features.data(), features.size(),
                input_shape.data(), input_shape.size());

            // Run inference
            const char* input_names[] = {input_name_.c_str()};
            const char* output_names[] = {output_name_.c_str()};

            auto output_tensors = session_->Run(
                Ort::RunOptions{nullptr},
                input_names, &input_tensor, 1,
                output_names, 1);

            // Get output
            float* output_data = output_tensors[0].GetTensorMutableData<float>();
            double predicted_return = static_cast<double>(output_data[0]);

            // Generate signal
            Prediction pred;
            pred.predicted_return = predicted_return;
            pred.confidence = std::min(1.0, std::abs(predicted_return) / 0.05); // 5% = max confidence

            if (predicted_return > signal_threshold_) {
                pred.direction = 1;
                pred.signal = "BUY";
            } else if (predicted_return < -signal_threshold_) {
                pred.direction = -1;
                pred.signal = "SELL";
            } else {
                pred.direction = 0;
                pred.signal = "HOLD";
            }

            return pred;

        } catch (const Ort::Exception& e) {
            spdlog::error("Prediction failed: {}", e.what());
            return {0.0, 0.0, 0, "HOLD"};
        }
    }

    /**
     * Get the prediction horizon (days ahead).
     */
    int prediction_horizon() const { return prediction_horizon_; }

    /**
     * Set the signal threshold for buy/sell decisions.
     */
    void set_signal_threshold(double threshold) { signal_threshold_ = threshold; }

private:
    bool load_config(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        try {
            nlohmann::json config;
            file >> config;

            sequence_length_ = config.value("sequence_length", 60);
            prediction_horizon_ = config.value("prediction_horizon", 5);

            spdlog::info("Config loaded: seq_len={}, horizon={}",
                        sequence_length_, prediction_horizon_);
            return true;
        } catch (const std::exception& e) {
            spdlog::error("Config parse error: {}", e.what());
            return false;
        }
    }

    bool load_norm_stats(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return false;

        try {
            nlohmann::json stats;
            file >> stats;

            norm_stats_.clear();
            for (auto& [key, value] : stats.items()) {
                NormStats ns;
                ns.mean = value.value("mean", 0.0);
                ns.std = value.value("std", 1.0);
                if (ns.std < 1e-8) ns.std = 1.0;
                norm_stats_[key] = ns;
            }

            spdlog::info("Loaded {} normalization stats", norm_stats_.size());
            return true;
        } catch (const std::exception& e) {
            spdlog::error("Norm stats parse error: {}", e.what());
            return false;
        }
    }

    /**
     * Compute all features for the sequence window.
     * Must match the features used during training!
     */
    std::vector<float> compute_features() {
        std::vector<float> all_features;
        all_features.reserve(sequence_length_ * num_features_);

        // We need sequence_length_ rows of features
        size_t start_idx = bar_history_.size() - sequence_length_;

        // Pre-compute indicators for the full history
        auto closes = get_close_prices();
        auto highs = get_high_prices();
        auto lows = get_low_prices();
        auto volumes = get_volumes();

        auto returns = compute_returns(closes);
        auto rsi = compute_rsi(closes, 14);
        auto [macd, signal, hist] = compute_macd(closes);
        auto [bb_upper, bb_mid, bb_lower] = compute_bollinger(closes, 20, 2.0);
        auto atr = compute_atr(highs, lows, closes, 14);
        auto obv = compute_obv(closes, volumes);

        // For each timestep in sequence
        for (size_t i = 0; i < sequence_length_; ++i) {
            size_t idx = start_idx + i;

            // Feature order must match training!
            // returns, log_returns, rsi, macd, macd_signal, macd_hist,
            // bb_upper, bb_mid, bb_lower, bb_width, bb_pct,
            // atr, atr_pct, obv_change,
            // momentum_5, momentum_10, momentum_20,
            // volatility_5, volatility_10, volatility_20,
            // volume_ratio

            double close = closes[idx];

            // Returns
            all_features.push_back(static_cast<float>(returns[idx]));
            all_features.push_back(static_cast<float>(std::log1p(returns[idx])));

            // RSI
            all_features.push_back(static_cast<float>(rsi[idx]));

            // MACD
            all_features.push_back(static_cast<float>(macd[idx]));
            all_features.push_back(static_cast<float>(signal[idx]));
            all_features.push_back(static_cast<float>(hist[idx]));

            // Bollinger Bands
            all_features.push_back(static_cast<float>(bb_upper[idx]));
            all_features.push_back(static_cast<float>(bb_mid[idx]));
            all_features.push_back(static_cast<float>(bb_lower[idx]));
            double bb_width = (bb_upper[idx] - bb_lower[idx]) / bb_mid[idx];
            double bb_pct = (bb_upper[idx] > bb_lower[idx]) ?
                (close - bb_lower[idx]) / (bb_upper[idx] - bb_lower[idx]) : 0.5;
            all_features.push_back(static_cast<float>(bb_width));
            all_features.push_back(static_cast<float>(bb_pct));

            // ATR
            all_features.push_back(static_cast<float>(atr[idx]));
            all_features.push_back(static_cast<float>(close > 0 ? atr[idx] / close : 0));

            // OBV change
            double obv_change = (idx > 0 && obv[idx-1] != 0) ?
                (obv[idx] - obv[idx-1]) / std::abs(obv[idx-1]) : 0;
            all_features.push_back(static_cast<float>(obv_change));

            // Momentum (5, 10, 20 days)
            all_features.push_back(static_cast<float>(momentum(closes, idx, 5)));
            all_features.push_back(static_cast<float>(momentum(closes, idx, 10)));
            all_features.push_back(static_cast<float>(momentum(closes, idx, 20)));

            // Volatility (5, 10, 20 days)
            all_features.push_back(static_cast<float>(volatility(returns, idx, 5)));
            all_features.push_back(static_cast<float>(volatility(returns, idx, 10)));
            all_features.push_back(static_cast<float>(volatility(returns, idx, 20)));

            // Volume ratio
            double avg_vol = rolling_mean(volumes, idx, 20);
            double vol_ratio = (avg_vol > 0) ? volumes[idx] / avg_vol : 1.0;
            all_features.push_back(static_cast<float>(vol_ratio));
        }

        return all_features;
    }

    void normalize(std::vector<float>& features) {
        // Feature names in order
        std::vector<std::string> feature_names = {
            "returns", "log_returns", "rsi", "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_mid", "bb_lower", "bb_width", "bb_pct",
            "atr", "atr_pct", "obv_change",
            "momentum_5", "momentum_10", "momentum_20",
            "volatility_5", "volatility_10", "volatility_20",
            "volume_ratio"
        };

        size_t nf = feature_names.size();
        for (size_t t = 0; t < sequence_length_; ++t) {
            for (size_t f = 0; f < nf; ++f) {
                size_t idx = t * nf + f;
                if (idx < features.size()) {
                    auto it = norm_stats_.find(feature_names[f]);
                    if (it != norm_stats_.end()) {
                        features[idx] = static_cast<float>(
                            (features[idx] - it->second.mean) / it->second.std);
                    }
                }
            }
        }
    }

    // Helper functions for extracting price arrays
    std::vector<double> get_close_prices() {
        std::vector<double> v;
        v.reserve(bar_history_.size());
        for (const auto& bar : bar_history_) v.push_back(bar.close);
        return v;
    }

    std::vector<double> get_high_prices() {
        std::vector<double> v;
        v.reserve(bar_history_.size());
        for (const auto& bar : bar_history_) v.push_back(bar.high);
        return v;
    }

    std::vector<double> get_low_prices() {
        std::vector<double> v;
        v.reserve(bar_history_.size());
        for (const auto& bar : bar_history_) v.push_back(bar.low);
        return v;
    }

    std::vector<double> get_volumes() {
        std::vector<double> v;
        v.reserve(bar_history_.size());
        for (const auto& bar : bar_history_) v.push_back(bar.volume);
        return v;
    }

    // Technical indicator calculations
    std::vector<double> compute_returns(const std::vector<double>& prices) {
        std::vector<double> ret(prices.size(), 0.0);
        for (size_t i = 1; i < prices.size(); ++i) {
            if (prices[i-1] > 0) {
                ret[i] = (prices[i] - prices[i-1]) / prices[i-1];
            }
        }
        return ret;
    }

    std::vector<double> compute_rsi(const std::vector<double>& prices, int period) {
        std::vector<double> rsi(prices.size(), 50.0);
        if (prices.size() < static_cast<size_t>(period + 1)) return rsi;

        double avg_gain = 0, avg_loss = 0;

        // Initial average
        for (int i = 1; i <= period; ++i) {
            double change = prices[i] - prices[i-1];
            if (change > 0) avg_gain += change;
            else avg_loss -= change;
        }
        avg_gain /= period;
        avg_loss /= period;

        for (size_t i = period; i < prices.size(); ++i) {
            if (i > static_cast<size_t>(period)) {
                double change = prices[i] - prices[i-1];
                if (change > 0) {
                    avg_gain = (avg_gain * (period - 1) + change) / period;
                    avg_loss = (avg_loss * (period - 1)) / period;
                } else {
                    avg_gain = (avg_gain * (period - 1)) / period;
                    avg_loss = (avg_loss * (period - 1) - change) / period;
                }
            }

            if (avg_loss < 1e-10) rsi[i] = 100.0;
            else {
                double rs = avg_gain / avg_loss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }
        return rsi;
    }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    compute_macd(const std::vector<double>& prices) {
        auto ema12 = compute_ema(prices, 12);
        auto ema26 = compute_ema(prices, 26);

        std::vector<double> macd(prices.size());
        for (size_t i = 0; i < prices.size(); ++i) {
            macd[i] = ema12[i] - ema26[i];
        }

        auto signal = compute_ema(macd, 9);

        std::vector<double> hist(prices.size());
        for (size_t i = 0; i < prices.size(); ++i) {
            hist[i] = macd[i] - signal[i];
        }

        return {macd, signal, hist};
    }

    std::vector<double> compute_ema(const std::vector<double>& data, int period) {
        std::vector<double> ema(data.size());
        if (data.empty()) return ema;

        double multiplier = 2.0 / (period + 1);
        ema[0] = data[0];

        for (size_t i = 1; i < data.size(); ++i) {
            ema[i] = (data[i] - ema[i-1]) * multiplier + ema[i-1];
        }
        return ema;
    }

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
    compute_bollinger(const std::vector<double>& prices, int period, double num_std) {
        std::vector<double> upper(prices.size()), mid(prices.size()), lower(prices.size());

        for (size_t i = 0; i < prices.size(); ++i) {
            if (i < static_cast<size_t>(period - 1)) {
                mid[i] = prices[i];
                upper[i] = prices[i];
                lower[i] = prices[i];
            } else {
                double sum = 0, sq_sum = 0;
                for (int j = 0; j < period; ++j) {
                    double p = prices[i - j];
                    sum += p;
                    sq_sum += p * p;
                }
                double mean = sum / period;
                double var = (sq_sum / period) - (mean * mean);
                double std_dev = std::sqrt(std::max(0.0, var));

                mid[i] = mean;
                upper[i] = mean + num_std * std_dev;
                lower[i] = mean - num_std * std_dev;
            }
        }
        return {upper, mid, lower};
    }

    std::vector<double> compute_atr(const std::vector<double>& highs,
                                     const std::vector<double>& lows,
                                     const std::vector<double>& closes,
                                     int period) {
        std::vector<double> atr(closes.size(), 0);
        if (closes.size() < 2) return atr;

        std::vector<double> tr(closes.size());
        tr[0] = highs[0] - lows[0];

        for (size_t i = 1; i < closes.size(); ++i) {
            double hl = highs[i] - lows[i];
            double hc = std::abs(highs[i] - closes[i-1]);
            double lc = std::abs(lows[i] - closes[i-1]);
            tr[i] = std::max({hl, hc, lc});
        }

        // Simple moving average for ATR
        for (size_t i = 0; i < closes.size(); ++i) {
            if (i < static_cast<size_t>(period - 1)) {
                atr[i] = tr[i];
            } else {
                double sum = 0;
                for (int j = 0; j < period; ++j) {
                    sum += tr[i - j];
                }
                atr[i] = sum / period;
            }
        }
        return atr;
    }

    std::vector<double> compute_obv(const std::vector<double>& closes,
                                     const std::vector<double>& volumes) {
        std::vector<double> obv(closes.size(), 0);
        if (closes.empty()) return obv;

        obv[0] = volumes[0];
        for (size_t i = 1; i < closes.size(); ++i) {
            if (closes[i] > closes[i-1]) {
                obv[i] = obv[i-1] + volumes[i];
            } else if (closes[i] < closes[i-1]) {
                obv[i] = obv[i-1] - volumes[i];
            } else {
                obv[i] = obv[i-1];
            }
        }
        return obv;
    }

    double momentum(const std::vector<double>& prices, size_t idx, int period) {
        if (idx < static_cast<size_t>(period)) return 0;
        if (prices[idx - period] <= 0) return 0;
        return (prices[idx] - prices[idx - period]) / prices[idx - period];
    }

    double volatility(const std::vector<double>& returns, size_t idx, int period) {
        if (idx < static_cast<size_t>(period)) return 0;

        double sum = 0, sq_sum = 0;
        for (int i = 0; i < period; ++i) {
            double r = returns[idx - i];
            sum += r;
            sq_sum += r * r;
        }
        double mean = sum / period;
        double var = (sq_sum / period) - (mean * mean);
        return std::sqrt(std::max(0.0, var));
    }

    double rolling_mean(const std::vector<double>& data, size_t idx, int period) {
        if (idx < static_cast<size_t>(period - 1)) return data[idx];
        double sum = 0;
        for (int i = 0; i < period; ++i) {
            sum += data[idx - i];
        }
        return sum / period;
    }

    // Member variables
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;

    std::string input_name_;
    std::string output_name_;

    size_t sequence_length_ = 60;
    size_t num_features_ = 21;
    int prediction_horizon_ = 5;
    double signal_threshold_ = 0.01;  // 1% return threshold

    std::deque<Bar> bar_history_;
    std::unordered_map<std::string, NormStats> norm_stats_;

    bool initialized_ = false;
};

} // namespace argus::ml

#endif // ARGUS_ONNX_ENABLED
