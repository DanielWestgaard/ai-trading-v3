# models/model_factory.py
class ModelFactory:
    """Factory for creating model instances."""
    
    @staticmethod
    def create_model(model_type, **kwargs):
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Parameters for the model
            
        Returns:
            Model instance
        """
        if model_type == 'random_forest':
            from models.tree.random_forest_model import RandomForestModel
            return RandomForestModel(**kwargs)
        
        elif model_type == 'xgboost':
            from models.tree.xgboost_model import XGBoostModel
            return XGBoostModel(**kwargs)
        
        # elif model_type == 'lstm':
        #     from models.deep.lstm_model import LSTMModel
        #     return LSTMModel(**kwargs)
            
        # elif model_type == 'linear':
        #     from models.linear.linear_model import LinearModel
        #     return LinearModel(**kwargs)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")