class Cte:
    # equation-types
    LINEAR = "linear"
    NONLINEAR = 'non-linear'
    NONADDITIVE = 'non-additive'

    # losss
    L2 = 'l2'
    L1 = 'l1'

    X1 = 'x1'

    # Node names for multisynth
    Y = 'Y'
    AI = 'AI'
    P = 'P'
    I = 'I'
    AP = 'AP'

    # Node names for adult
    RACE = 'race'
    AGE = 'age'
    NATIVE_COUNTRY = 'native_country'
    GENDER = 'gender'

    EDU = 'edu'
    HOUR = 'hour'
    WORK_CLASS = 'work_class'
    MARITIAL = 'maritial'

    OCCUPATION = 'occupation'
    RELATIONSHIP = 'relationship'
    INCOME = 'income'

    DURATION = 'duration'
    SAVINGS = 'savings'
    LOAN_AMOUNT = 'loan_amount'

    # fairness
    NONE = "none"
    DP = "dp"
    EOP = "eop"

    # Datasets
    SENS = 'sex'  # sensitive attribute for CF fairness

    TRIANGLESENS = 'trianglesens'
    COLLIDERSENS = 'collidersens'
    CHAINSENS = 'chainsens'
    MULTISYNTH = 'multisynth'
    TOY = 'toy'

    LOAN = 'loan'
    ADULT = 'adult'
    # MGRAPH = 'mgraph'
    GERMAN = 'german'

    DATASET_LIST = [
        COLLIDERSENS,
        TRIANGLESENS,
        LOAN,
        CHAINSENS,
        ADULT,
        GERMAN,
        MULTISYNTH]
    DATASET_LIST_TOY = [
        LOAN,
        ADULT,
        COLLIDERSENS,
        TRIANGLESENS,
        CHAINSENS,
        MULTISYNTH
    ]

    # Distribution
    BETA = 'beta'
    CONTINOUS_BERN = 'cb'
    BERNOULLI = 'ber'
    GAUSSIAN = 'normal'
    CATEGORICAL = 'cat'
    EXPONENTIAL = 'exp'
    DELTA = 'delta'

    # VACA Models
    VACA = 'vaca'
    VACA_PIWAE = 'vaca_piwae'

    # clf Models
    CLF_NN = 'nn'
    CLF_LIN = 'linear'

    # Optimizers
    ADAM = 'adam'
    RADAM = 'radam'
    ADAGRAD = 'adag'
    ADADELTA = 'adad'
    RMS = 'rms'
    ASGD = 'asgd'

    # Scheduler
    STEP_LR = 'step_lr'
    EXP_LR = 'exp_lr'

    # Activation
    TANH = 'tanh'
    RELU = 'relu'
    RELU6 = 'relu6'
    SOFTPLUS = 'softplus'
    RRELU = 'rrelu'
    LRELU = 'lrelu'
    ELU = 'elu'
    SELU = 'selu'
    SIGMOID = 'sigmoid'
    GLU = 'glu'
    IDENTITY = 'identity'
