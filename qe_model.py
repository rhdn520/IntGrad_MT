from comet import download_model, load_from_checkpoint

class CometQE:
    def __init__(self, withRef=True) -> None:
        if withRef:
            self.comet_path = download_model("Unbabel/XCOMET-XXL")
        else:
            self.comet_path = download_model("Unbabel/wmt23-cometkiwi-da-xxl")

        self.comet = load_from_checkpoint(self.comet_path)

    def predict(self, samples, batch_size=8, gpus=1):
        prediction = self.comet.predict(samples, batch_size=batch_size, gpus=gpus)
        return prediction.scores

