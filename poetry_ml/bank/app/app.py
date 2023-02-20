from io import StringIO
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import pandas as pd
import os
import joblib
from bank.feature import categorical_feature_mapper, numerical_feature_mapper

model = joblib.load('output/model.pickle')
app = FastAPI()


@app.post("/api/v1/predict/")
def predict(file: UploadFile = File(...)):
    """
    predict for a given dataset
    """
    input_data = pd.read_csv(file.file, delimiter=";").rename(columns=str.lower)
    input_data = input_data.loc[1:1000,:]
    
    # Process data. 
    categorical_feature = categorical_feature_mapper(input_data)
    numerical_feature = numerical_feature_mapper(input_data)
    df_mapped = pd.concat([numerical_feature, categorical_feature], axis=1)

    # Create predictions. 
    y_pred = model.predict_proba(df_mapped)

    # Combine predictions with class names and animal name.
    classes = model.classes_.tolist()
    proba_df = pd.DataFrame(y_pred, columns=classes)
    predictions = pd.concat([input_data,proba_df], axis=1)
    response = _convert_df_to_response(predictions)
    return response


def _convert_df_to_response(df: pd.DataFrame) -> StreamingResponse:
    """Convert a DataFrame to CSV response."""
    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    return response
