# Dummy class for warprnnt_pytorch
class RNNTLoss:
    def __init__(self, *args, debug=False, **kwargs):
        self.debug = debug
        if self.debug:
            print ("Dummy RNNTLoss constructor called.")
            print ("\targs: ", args)
            print ("\tkwargs: ", kwargs)

    def __call__(self, *args, **kwargs):
        if self.debug:
            print ("Dummy RNNTLoss function called.")
            print ("\targs: ", args)
            print ("\tkwargs: ", kwargs)
            print ("\treturns: 0.0")
        return 0.0
