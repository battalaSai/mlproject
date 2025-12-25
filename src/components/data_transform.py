import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_ob_file_path=os.path.join('artifact',"prepocessor.pkl")
    


class DataTranformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_trasformaer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scalor", StandardScaler())
                ]
            )
            logging.info("Numerical columns scaling completed")

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),
                    ("one hot encoder", OneHotEncoder()),
                    ("Standard Scaling", StandardScaler())
                ]
            )

            logging.info("Catergorical columns encoding completed")

            preprocessing = ColumnTransformer([
                ("Numerical Pipeline", num_pipeline),
                ("Categorical Pipeline", cat_pipeline)
            ])

            logging.info("Preprocesing completed")

            return preprocessing


        except Exception as e:
            CustomException(e,sys)

    def intiate_data_transform(self,train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Train and Testing data complete")
            logging.info("PObtaining Pre procesing object")
            preprocessing_obj = self.get_data_trasformaer_object()

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying Preprocessing object on train data")
            input_feature_train_array=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array=preprocessing_obj.transform(input_feature_test_df)
            train_arr=np.c_[
                input_feature_train_array,np.array(input_feature_train_array)
            ]
            test_arr=np.c_[
                input_feature_test_array,np.array(input_feature_test_array)
            ]
            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )


        except Exception as e:
            CustomException(e,sys)