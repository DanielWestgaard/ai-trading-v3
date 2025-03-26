# Model Management Best Practices

## Model Naming Convention
```
{market}_{timeframe}_{model_type}_{strategy}_{version}_{date}.model
```

Example:
```
eurusd_5m_xgb_trend_v1_20250325.model
us100_15m_lstm_breakout_v2_20250401.model
```

## Model Metadata File (JSON)
Every model should have an accompanying metadata file:

```json
{
  "model_id": "eurusd_5m_xgb_trend_v1",
  "created_at": "2025-03-25T14:30:00Z",
  "author": "your_name",
  "framework": "xgboost",
  "version": "1.5.1",
  "features": ["rsi_14", "ema_20", "ema_50", "atr_14", "hour_of_day"],
  "hyperparameters": {
    "max_depth": 5,
    "learning_rate": 0.01,
    "n_estimators": 100
  },
  "training_period": {
    "start": "2023-01-01",
    "end": "2024-12-31"
  },
  "performance": {
    "backtest": {
      "sharpe_ratio": 1.85,
      "max_drawdown": 0.12,
      "win_rate": 0.58,
      "profit_factor": 1.75
    },
    "validation": {
      "sharpe_ratio": 1.62,
      "max_drawdown": 0.15,
      "win_rate": 0.54,
      "profit_factor": 1.68
    }
  },
  "status": "production",
  "notes": "Initial trend-following model for EURUSD"
}
```

## Model Registry Utility

Create a `model_registry.py` file to help with model management:

```python
import os
import json
import pandas as pd
from datetime import datetime

class ModelRegistry:
    def __init__(self, storage_root="./model_storage"):
        self.storage_root = storage_root
        self.registry_file = os.path.join(storage_root, "metadata", "registry.json")
        os.makedirs(os.path.dirname(self.registry_file), exist_ok=True)
        self._load_registry()
    
    def _load_registry(self):
        if os.path.exists(self.registry_file):
            with open(self.registry_file, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {
                "models": []
            }
    
    def save_model(self, model, model_info, environment="staging"):
        """
        Save a model and its metadata to the appropriate location
        
        Args:
            model: The trained model object
            model_info: Dict containing model metadata
            environment: Where to save ("staging", "production", "archive")
        """
        # Create model_id if not provided
        if "model_id" not in model_info:
            model_id = f"{model_info.get('market', 'unknown')}_{model_info.get('timeframe', '0m')}_{model_info.get('model_type', 'model')}_{model_info.get('strategy', 'unknown')}_v{model_info.get('version', '1')}"
            model_info["model_id"] = model_id
        
        # Add timestamp
        model_info["created_at"] = datetime.now().isoformat()
        
        # Determine save path
        market = model_info.get("market", "unknown")
        version = f"v{model_info.get('version', '1')}"
        
        save_dir = os.path.join(
            self.storage_root, 
            environment,
            market,
            version
        )
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_filename = f"{model_info['model_id']}_{datetime.now().strftime('%Y%m%d')}.model"
        model_path = os.path.join(save_dir, model_filename)
        
        # Save based on model type
        if hasattr(model, 'save_model'):
            model.save_model(model_path)
        else:
            import joblib
            joblib.dump(model, model_path)
        
        # Save metadata
        meta_path = os.path.join(save_dir, f"{model_info['model_id']}_{datetime.now().strftime('%Y%m%d')}.json")
        with open(meta_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        # Add to registry
        model_info["file_path"] = model_path
        model_info["metadata_path"] = meta_path
        model_info["environment"] = environment
        
        self.registry["models"].append(model_info)
        self._save_registry()
        
        return model_path, meta_path
    
    def load_model(self, model_id, version=None, environment=None):
        """
        Load a model by its ID
        
        Args:
            model_id: The ID of the model to load
            version: Specific version to load (optional)
            environment: Which environment to load from (optional)
        
        Returns:
            model, model_info
        """
        # Filter models
        candidates = [m for m in self.registry["models"] if m["model_id"].startswith(model_id)]
        
        if version:
            candidates = [m for m in candidates if m.get("version") == version]
        
        if environment:
            candidates = [m for m in candidates if m.get("environment") == environment]
        
        if not candidates:
            raise ValueError(f"No model found with ID {model_id}")
        
        # Get the latest model
        model_info = sorted(candidates, key=lambda x: x.get("created_at", ""), reverse=True)[0]
        
        # Load model based on file extension
        model_path = model_info["file_path"]
        
        if model_path.endswith('.model'):
            import xgboost as xgb
            model = xgb.Booster()
            model.load_model(model_path)
        else:
            import joblib
            model = joblib.load(model_path)
        
        return model, model_info
    
    def list_models(self, environment=None, market=None, status=None):
        """List models matching specified criteria"""
        models = self.registry["models"]
        
        if environment:
            models = [m for m in models if m.get("environment") == environment]
        
        if market:
            models = [m for m in models if m.get("market") == market]
        
        if status:
            models = [m for m in models if m.get("status") == status]
        
        return pd.DataFrame(models)
    
    def promote_model(self, model_id, to_environment="production"):
        """Promote a model to a new environment (e.g., staging to production)"""
        for i, model in enumerate(self.registry["models"]):
            if model["model_id"] == model_id:
                # Copy model to new location
                source_path = model["file_path"]
                meta_path = model["metadata_path"]
                
                # Load metadata
                with open(meta_path, 'r') as f:
                    model_info = json.load(f)
                
                # Determine new paths
                market = model_info.get("market", "unknown")
                version = f"v{model_info.get('version', '1')}"
                
                target_dir = os.path.join(
                    self.storage_root, 
                    to_environment,
                    market,
                    version
                )
                
                os.makedirs(target_dir, exist_ok=True)
                
                # Copy files
                import shutil
                new_model_path = os.path.join(target_dir, os.path.basename(source_path))
                new_meta_path = os.path.join(target_dir, os.path.basename(meta_path))
                
                shutil.copy2(source_path, new_model_path)
                shutil.copy2(meta_path, new_meta_path)
                
                # Update registry
                self.registry["models"][i]["environment"] = to_environment
                self.registry["models"][i]["file_path"] = new_model_path
                self.registry["models"][i]["metadata_path"] = new_meta_path
                
                self._save_registry()
                return True
        
        return False
    
    def _save_registry(self):
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
```

## Workflow for Model Management

1. **During Development:**
   - Create models in notebooks
   - Save experimental models to `model_storage/staging/`

2. **After Validation:**
   - Run thorough evaluation on staging models
   - Promote successful models to production

3. **Versioning:**
   - Increment versions for significant changes
   - Archive outdated models instead of deleting

4. **Demo Testing:**
   - Always use models from `production/` for demo trading
   - Log performance metrics back to model metadata

5. **Performance Tracking:**
   - Create performance dashboards using metadata
   - Compare model versions and strategies