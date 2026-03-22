import sys
from pathlib import Path
sys.path.insert(0, '.')

from app.ml.models.acmi import ACMIPredictor

def test_inference():
    print("=" * 60)
    print("TESTING ACMI++ PREDICTOR")
    print("=" * 60)
    
    model_path = "a:/Project/stock-market-prediction/models/acmi_best_model.pt"
    scaler_path = "a:/Project/stock-market-prediction/models/scalers.pkl"
    ensemble_path = "a:/Project/stock-market-prediction/models/acmi_ensemble.pt"
    
    try:
        predictor = ACMIPredictor(
            model_path=model_path,
            scaler_path=scaler_path,
            ensemble_path=ensemble_path,
            device="cpu"
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    symbol = "AAPL"
    print(f"\nRunning prediction for {symbol}...")
    try:
        result = predictor.predict(symbol)
        print("Prediction successful!")
        print(f"Latest Price: {result['latest_price']}")
        print(f"Regime: {result['regime']}")
        print(f"Crash Prob: {result['crash_prob']:.2%}")
        for h in result["horizons"]:
            print(f"Horizon {h}: Point {result['horizons'][h]['point']:.4f}, Interval {result['horizons'][h]['interval']}")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__ == "__main__":
    test_inference()
