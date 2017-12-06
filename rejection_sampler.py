import numpy as np


def rejection_sample(pdf,xmin,xmax,pdfmax,no_samples):
    """

    :param pdf: function to evaluate pdf
    :param no_samples: number of samples to generate.
    :return: generated sample vector.
    """
    samples = []
    samples_superSet = np.random.uniform(xmin,xmax,no_samples)
    try:
        samples_proposalValue = np.random.uniform(0,pdfmax,no_samples)
    except OverflowError:
        pass

    pdf_vals = pdf(samples_superSet)
    final_samples = samples_superSet[samples_proposalValue<pdf_vals]

    return final_samples





