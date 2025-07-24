# GenEval: A Unified Evaluation Framework for Generative AI Applications

A unified evaluation framework that provides a standardized interface for evaluating generative AI models across different frameworks like RAGAS and DeepEval.

## Project Structure

```
gen-eval/
├── pyproject.toml              
├── README.md                  
├── LICENSE                     
├── .gitignore                  
├── main.py                     
├── geneval/                    
│   ├── schemas.py             
│   ├── normalization.py       
│   ├── framework.py           
│   └── adapters/              
│       ├── __init__.py        
│       ├── ragas_adapter.py   
│       └── deepeval_adapter.py 
└── tests/                     
    └── test_framework.py      
```

