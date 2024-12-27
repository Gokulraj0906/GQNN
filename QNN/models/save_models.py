import pickle

class SaveModel:
    """
    A utility class to save and load models.
    """

    @staticmethod
    def save_model(model, file_path: str):
        """
        Save a model to a specified file path.

        Args:
            model: The model object to be saved.
            file_path (str): Path to save the model file.
        """
        try:
            with open(file_path, 'wb') as file:
                pickle.dump(model, file)
            print(f"Model saved successfully at {file_path}")
        except Exception as e:
            print(f"Failed to save the model: {e}")

    @staticmethod
    def load_model(file_path: str):
        """
        Load a model from a specified file path.

        Args:
            file_path (str): Path to the model file.

        Returns:
            The loaded model object.
        """
        try:
            with open(file_path, 'rb') as file:
                model = pickle.load(file)
            print(f"Model loaded successfully from {file_path}")
            return model
        except Exception as e:
            print(f"Failed to load the model: {e}")
            return None

