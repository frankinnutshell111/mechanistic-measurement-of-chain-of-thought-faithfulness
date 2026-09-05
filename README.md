The write out for this project is here: 

https://docs.google.com/document/d/1KAUza1pJM7N2t0euB7xwUGlxzSoAsa7p9CFqizMOAP4/edit?usp=sharing

1, Generate the effective dataset with

    python -m scripts.generate_paired_dataset

    select the slice of data and the hinting method in the scripts

2, Run CoT faithfulness analysis with 

    python -m scripts.cot_faithfulness_measure

    choose the data id and layer numbers

3. Calculate the (T,D) values with

    python -m scripts.score_calculation

4. Interprete the result with either
    
    python -m scripts.metric

    or

    python -m scripts.l2_metric


