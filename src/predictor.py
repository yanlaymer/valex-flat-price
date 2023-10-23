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
        logger.info(f"PRICE PER SQUARE METERS (WITHOUT CORRECTION): {prediction}")
        
        analog_min_price = self.model_entry['analog_prices_min'] * self.entry['total_square']
        analog_median_price = self.model_entry['analog_prices_median'] * self.entry['total_square']
        analog_max_price = self.model_entry['analog_prices_max'] * self.entry['total_square']

        # Determine the best price based on the conditions
        corrected_price = max(
            prediction * self.entry['total_square'],
            analog_min_price if analog_min_price > prediction else 0,
            analog_median_price if analog_median_price * 1.5 > prediction else 0,
            analog_max_price if analog_max_price * 1.5 < prediction else 0
        )
        
        logger.info(f"MIN PRICE: {analog_min_price}")
        logger.info(f"MEDIAN PRICE: {analog_median_price}")
        logger.info(f"MAX PRICE: {analog_max_price}")
        
        if corrected_price == 0:
            corrected_price = analog_median_price * self.entry['total_square']

        # Log the correction applied
        if corrected_price == analog_min_price:
            logger.info("CORRECTION BY MIN")
        elif corrected_price == analog_median_price:
            logger.info("CORRECTION BY MEDIAN")
        elif corrected_price == analog_max_price:
            logger.info("CORRECTION BY MAX")
            

        self.final_price = corrected_price

        
        logger.info(f"PRICE(AFTER CORRECTION): {corrected_price}")
        logger.info(f"PRICE PER SQUARE METERS (AFTER CORRECTION): {corrected_price / self.entry['total_square']}")
        analog_links = [self.model_entry['analog_1'], self.model_entry['analog_2'], self.model_entry['analog_3']]
        
        return self.final_price, analog_links
        
        
        
        
        
    

    
    
    
            