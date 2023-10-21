import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from loguru import logger
from src.create_features import get_flat_features
from src.constants import MODEL_COLUMNS


class Predictor:
    def __init__(self, entry: pd.Series):
        self.entry = entry
        self.model = CatBoostRegressor().load_model("model/regression_flat.cbm")
        self.pred_price = None
        self.final_price = None
        
    def predict_price(self):
        self.model_entry = get_flat_features(self.entry)
        
        prediction = self.model.predict(self.model_entry[MODEL_COLUMNS])
        self.pred_price = prediction
        logger.info(f"PRICE (WITHOUT CORRECTION): {prediction * self.entry['total_square']}")
        
        if self.model_entry['analog_prices_min'] > prediction:
            logger.info("CORRECTION BY MIN")
            prediction = self.model_entry['analog_prices_min'] * self.entry['total_square']
        if self.model_entry['analog_prices_median'] * 1.5 > prediction:
            logger.info("CORRECTION BY MEDIAN")
            prediction = self.model_entry['analog_prices_median'] * self.entry['total_square']
        if self.model_entry['analog_prices_max'] * 1.5 > prediction:
            logger.info("CORRECTION BY MAX")
            prediction = self.model_entry['analog_prices_max'] * self.entry['total_square']
            
        self.final_price = prediction
        
        if self.final_price == self.pred_price:
            logger.info("NO CORRECTION")
        
        logger.info(f"PRICE(AFTER CORRECTION): {prediction}")
        analog_links = [self.model_entry['analog_1'], self.model_entry['analog_2'], self.model_entry['analog_3']]
        
        return prediction, analog_links
        
        
        
        
        
    

    
    
    
            