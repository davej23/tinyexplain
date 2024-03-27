# General Usage

## **Create explainer**
Specify a model and the task type that corresponds to the model.
```python
from tinyexplain.explain import [EXPLAINER]

explainer = EXPLAINER(model, task, **kwargs)
```

## **Selecting a postprocessing function**
This function will do any postprocessing that is required on the raw model output to produce a tensor that is ready to be evaluated against the ground truth. In a lot of cases, the identity function can be used `lambda x: x`.

## **Selecting a score function**
This function will evaluate the output of the postprocessing function against the ground truth target. This will yield a 'score' for the model prediction which will be used to produce the explanation.

## **Run the explainer**
```python
explanation = explainer.explain(inputs, targets, postprocessing_fn, score_fn)
```
