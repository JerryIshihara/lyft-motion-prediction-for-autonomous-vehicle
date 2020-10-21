import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-model", "--model", help="Model name")
parser.add_argument("-d", "--debug", help="Debug mode", action='store_true')
parser.add_argument("-gpu", "--gpu", help="Training on GPU",
                    action='store_true')
args = parser.parse_args()


class ModelNotFoundException(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # Now for your custom code...


try:
    raise ModelNotFoundException(
        "ModelNotFoundException:\n  Unable to find the mode: {}".format(args.model))
except ModelNotFoundException as e:
    print(e)

print("model {} debug {} platform {}".format(
    args.model,
    args.debug,
    args.gpu,
))
