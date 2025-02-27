class QiskitServiceManager:
    def __init__(self):
        self.provider = None

    def save_account(self, api_key: str, channel: str, instance:str,**kwargs):
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            QiskitRuntimeService.save_account(token=api_key, channel=channel,instance=instance,**kwargs)
            self.provider = QiskitRuntimeService()
            return "Account is saved successfully."
        except Exception as e:
            return f"Error saving account: {e}"

    def load_account(self):
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            self.provider = QiskitRuntimeService()
            return "Account is loaded successfully."
        except Exception as e:
            return f"Error loading account: {e}"