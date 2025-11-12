import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier, HistGradientBoostingRegressor,
    RandomForestClassifier, RandomForestRegressor
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class EnhancedEnsembleModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scalers = {}
        self.models = {}
        self.feature_importance = {}
        
    def _create_base_models(self):
        """Create diverse base models for ensemble"""
        models = {
            'hgb_cls': HistGradientBoostingClassifier(
                max_depth=8, learning_rate=0.03, max_iter=500, 
                l2_regularization=0.1, random_state=self.random_state
            ),
            'rf_cls': RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=self.random_state
            ),
            'lr_cls': LogisticRegression(
                C=0.1, max_iter=1000, random_state=self.random_state
            ),
            'hgb_reg': HistGradientBoostingRegressor(
                max_depth=8, learning_rate=0.03, max_iter=500,
                l2_regularization=0.1, random_state=self.random_state
            ),
            'rf_reg': RandomForestRegressor(
                n_estimators=200, max_depth=10, min_samples_split=10,
                min_samples_leaf=5, random_state=self.random_state
            ),
            'ridge_reg': Ridge(alpha=1.0, random_state=self.random_state)
        }
        return models
    
    def _optimize_hyperparameters(self, X, y_cls, y_reg):
        """Optimize hyperparameters using time series cross-validation"""
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Optimize HGB classifier
        hgb_cls_params = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.02, 0.03, 0.05],
            'l2_regularization': [0.05, 0.1, 0.2]
        }
        
        hgb_cls = HistGradientBoostingClassifier(max_iter=300, random_state=self.random_state)
        hgb_cls_search = GridSearchCV(hgb_cls, hgb_cls_params, cv=tscv, scoring='roc_auc', n_jobs=-1)
        hgb_cls_search.fit(X, y_cls)
        
        # Optimize HGB regressor
        hgb_reg_params = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.02, 0.03, 0.05],
            'l2_regularization': [0.05, 0.1, 0.2]
        }
        
        hgb_reg = HistGradientBoostingRegressor(max_iter=300, random_state=self.random_state)
        hgb_reg_search = GridSearchCV(hgb_reg, hgb_reg_params, cv=tscv, scoring='r2', n_jobs=-1)
        hgb_reg_search.fit(X, y_reg)
        
        return hgb_cls_search.best_estimator_, hgb_reg_search.best_estimator_
    
    def train(self, train_df, feature_cols):
        """Train enhanced ensemble model"""
        X = train_df[feature_cols].astype(float)
        y_cls = train_df["y_up_3d"].astype(int)
        y_reg = train_df["fwd_ret_3d"].astype(float)
        
        # Remove outliers for better training
        q99 = y_reg.quantile(0.99)
        q01 = y_reg.quantile(0.01)
        mask = (y_reg >= q01) & (y_reg <= q99)
        X, y_cls, y_reg = X[mask], y_cls[mask], y_reg[mask]
        
        # Scale features
        self.scalers['standard'] = StandardScaler()
        X_scaled = self.scalers['standard'].fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        
        # Optimize key models
        print("🔧 Optimizing hyperparameters...")
        best_hgb_cls, best_hgb_reg = self._optimize_hyperparameters(X_scaled, y_cls, y_reg)
        
        # Create ensemble
        base_models = self._create_base_models()
        base_models['hgb_cls'] = best_hgb_cls
        base_models['hgb_reg'] = best_hgb_reg
        
        # Train all models with time series CV
        tscv = TimeSeriesSplit(n_splits=4)
        cv_scores = {'auc': [], 'acc': [], 'r2': []}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_cls_train, y_cls_val = y_cls.iloc[train_idx], y_cls.iloc[val_idx]
            y_reg_train, y_reg_val = y_reg.iloc[train_idx], y_reg.iloc[val_idx]
            
            # Train classifiers
            for name in ['hgb_cls', 'rf_cls', 'lr_cls']:
                base_models[name].fit(X_train, y_cls_train)
            
            # Train regressors
            for name in ['hgb_reg', 'rf_reg', 'ridge_reg']:
                base_models[name].fit(X_train, y_reg_train)
            
            # Ensemble predictions
            cls_preds = []
            for name in ['hgb_cls', 'rf_cls', 'lr_cls']:
                if hasattr(base_models[name], 'predict_proba'):
                    cls_preds.append(base_models[name].predict_proba(X_val)[:, 1])
                else:
                    cls_preds.append(base_models[name].decision_function(X_val))
            
            reg_preds = []
            for name in ['hgb_reg', 'rf_reg', 'ridge_reg']:
                reg_preds.append(base_models[name].predict(X_val))
            
            # Weighted ensemble (give more weight to best performing models)
            ensemble_cls_pred = 0.5 * cls_preds[0] + 0.3 * cls_preds[1] + 0.2 * cls_preds[2]
            ensemble_reg_pred = 0.5 * reg_preds[0] + 0.3 * reg_preds[1] + 0.2 * reg_preds[2]
            
            # Calculate metrics
            cv_scores['auc'].append(roc_auc_score(y_cls_val, ensemble_cls_pred))
            cv_scores['acc'].append(accuracy_score(y_cls_val, (ensemble_cls_pred >= 0.5).astype(int)))
            cv_scores['r2'].append(r2_score(y_reg_val, ensemble_reg_pred))
        
        # Train final models on all data
        for name, model in base_models.items():
            model.fit(X_scaled, y_cls if 'cls' in name else y_reg)
        
        self.models = base_models
        
        # Calculate feature importance
        self._calculate_feature_importance(feature_cols)
        
        metrics = {
            "auc": float(np.mean(cv_scores['auc'])),
            "acc": float(np.mean(cv_scores['acc'])),
            "r2": float(np.mean(cv_scores['r2'])),
            "n": int(len(X)),
            "auc_std": float(np.std(cv_scores['auc'])),
            "acc_std": float(np.std(cv_scores['acc'])),
            "r2_std": float(np.std(cv_scores['r2']))
        }
        
        return metrics
    
    def _calculate_feature_importance(self, feature_cols):
        """Calculate ensemble feature importance"""
        importance_scores = {}
        
        # Get importance from tree-based models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                for i, feat in enumerate(feature_cols):
                    if feat not in importance_scores:
                        importance_scores[feat] = []
                    importance_scores[feat].append(model.feature_importances_[i])
        
        # Average importance across models
        self.feature_importance = {
            feat: np.mean(scores) for feat, scores in importance_scores.items()
        }
    
    def predict(self, X, feature_cols):
        """Make ensemble predictions"""
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scalers['standard'].transform(X[feature_cols])
            X_scaled = pd.DataFrame(X_scaled, columns=feature_cols, index=X.index)
        else:
            X_scaled = self.scalers['standard'].transform(X)
        
        # Classification ensemble
        cls_preds = []
        for name in ['hgb_cls', 'rf_cls', 'lr_cls']:
            if hasattr(self.models[name], 'predict_proba'):
                cls_preds.append(self.models[name].predict_proba(X_scaled)[:, 1])
            else:
                cls_preds.append(self.models[name].decision_function(X_scaled))
        
        # Regression ensemble
        reg_preds = []
        for name in ['hgb_reg', 'rf_reg', 'ridge_reg']:
            reg_preds.append(self.models[name].predict(X_scaled))
        
        # Weighted ensemble
        p_up = 0.5 * cls_preds[0] + 0.3 * cls_preds[1] + 0.2 * cls_preds[2]
        exp_ret = 0.5 * reg_preds[0] + 0.3 * reg_preds[1] + 0.2 * reg_preds[2]
        
        return p_up, exp_ret
    
    def get_prediction_confidence(self, X, feature_cols):
        """Calculate prediction confidence based on model agreement"""
        if isinstance(X, pd.DataFrame):
            X_scaled = self.scalers['standard'].transform(X[feature_cols])
        else:
            X_scaled = self.scalers['standard'].transform(X)
        
        # Get predictions from all models
        cls_preds = []
        for name in ['hgb_cls', 'rf_cls', 'lr_cls']:
            if hasattr(self.models[name], 'predict_proba'):
                cls_preds.append(self.models[name].predict_proba(X_scaled)[:, 1])
            else:
                cls_preds.append(self.models[name].decision_function(X_scaled))
        
        reg_preds = []
        for name in ['hgb_reg', 'rf_reg', 'ridge_reg']:
            reg_preds.append(self.models[name].predict(X_scaled))
        
        # Calculate agreement (inverse of standard deviation)
        cls_agreement = 1 / (1 + np.std(cls_preds, axis=0))
        reg_agreement = 1 / (1 + np.std(reg_preds, axis=0))
        
        return cls_agreement, reg_agreement

def train_enhanced_models(train_df, feature_cols):
    """Train enhanced ensemble models"""
    model = EnhancedEnsembleModel()
    metrics = model.train(train_df, feature_cols)
    return model, metrics