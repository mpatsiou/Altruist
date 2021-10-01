# Altruist Bot
The widespread use of Interpretable Machine Learning models accommodate the inherent human curiosity both by providing intelligible evidence of “how” machine learning models work and “why” they are led to certain decisions. Nowadays, however, most Artificial Intelligence systems are considered more of black boxes, raising serious concerns about key ethical and societal issues. Therefore, a lot of work in academic research has been focused on Explainable Artificial Intelligence, and Interpretable Machine Learning, which make systems more transparent providing explanations about their decisions, but also information regarding
the following process. Among the problems that arise during the research are the lack of evaluation criteria of these explanations, as well as the difficulties that users face to comprehend them. Addressing these two main issues in the current thesis, we have designed a rule-based and user-friendly chatbot that uses the Altruist meta-explanation methodology while selecting explanations between multiple techniques that focus on feature importance contribution.

# Altruist
Altruist: Argumentative Explanations through Local Interpretations of Predictive Models

This paper introduces the "Altruist" (Argumentative expLanaTions thRoUgh local InterpretationS of predicTive models) method for transforming FI interpretations of ML models into insightful and validated explanations using argumentation based on classical logic. Altruist provides the local maximum truthful interpretation, as well as reasons for the truthfulness justification, and can be used as an easy-to-choose tool between X number of different interpretation techniques based on a few specific criteria. Altruist has innate virtues such as truthfulness, transparency and user-friendliness that characterise it as an apt tool for the XAI community.

## Altruist / ˈæl tru ɪst /
a person unselfishly concerned for or devoted to the welfare of others (opposed to egoist).

## Instructions
Please ensure you have docker installed on your desktop. Then:
```bash
docker pull johnmollas/altruist
```
After succesfully installing Altruist, please do:
```bash
docker run -p 8888:8888 johnmollas/altruist
```
Then, in your terminal copy the localhost url and open it in your browser.

## Contributors on Altruist Bot
Name | Email | Contribution
--- | --- | ---

[Thanos Mpatsioulas] | mpatsiou@csd.auth.gr | Main
[Ioannis Mollas](https://intelligence.csd.auth.gr/people/ioannis-mollas/) | iamollas@csd.auth.gr | Supervision
[Nick Bassiliades](https://intelligence.csd.auth.gr/people/bassiliades/) | nbassili@csd.auth.gr | Supervision

## See our Work
[LionLearn Interpretability Library](https://github.com/intelligence-csd-auth-gr/LionLearn) containing: 
1. [LioNets](https://github.com/iamollas/LionLearn/tree/master/LioNets): Local Interpretation Of Neural nETworkS through penultimate layer decoding
2. [LionForests](https://github.com/iamollas/LionLearn/tree/master/LionForests): Local Interpretation Of raNdom FORESts through paTh Selection

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
