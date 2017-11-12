from mlmodels.predictors import WineQuality


def train_local():
    predictor = WineQuality(version=1)

    predictor.train()
    predictor.save()


if __name__ == '__main__':
    train_local()
