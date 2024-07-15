# Finance Bot:

Leveraging the knowledge gained from the two sessions conducted by the Intel Unnati team, and through the execution of the provided Jupyter Notebooks, we have developed a another chatbot.

This chatbot has been fine-tuned on the [finance_alpaca.json](https://github.com/JoyM268/Intel-Unnati-Industrial-Training-Program-2024/blob/main/finance%20chatbot/finance_alpaca.json) dataset, specifically designed to address user inquiries which are finance related. The underlying architecture utilizes the [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v0.1) pre-trained model. 

The code and its output are present in the Jupyter Notebook [finance_chatbot.ipynb](https://github.com/JoyM268/Intel-Unnati-Industrial-Training-Program-2024/blob/main/finance%20chatbot/finance_chatbot.ipynb). The notebook can be executed easily by uploading it on [Kaggle](https://www.kaggle.com/) or [Google Colab](https://colab.research.google.com/).

While executing the notebook on **Kaggle**, store the access token as a secret in Kaggle with the label `Intel`. Also make sure that the internet option in Kaggle is Toggled on and the Accelerator is selected as `GPU P100` before executing the notebook.

If executing the notebook in **Google Colab**, make sure to change the file paths in the code and to store the access token as a secret in Colab with the name `Intel`. The Hugging Face login code must be changed to:
```python
from google.colab import userdata
from huggingface_hub import login
secret_value_0 = userdata.get('Intel')
login(token=secret_value_0)
``` 

Also make sure to change the runtime type to `T4 GPU`.

For Prompt:
```
what is meant by cost of equity?
```

The fine-tuned model responds:
```
Cost of equity is a financial metric used to evaluate the cost of capital for a company. It is calculated by dividing the total return on equity (ROE) by the cost of equity (COE). The COE is a measure of the cost of capital for a company, which is the amount of money required to finance the company's operations.

The cost of equity is important for investors because it provides a measure of the risk associated with investing in a company. It takes into account the cost of capital, which includes interest rates, taxes, and other expenses associated with investing in the company. By comparing the cost of equity to the ROE, investors can determine whether the company's investment is profitable or not.

The cost of equity is typically calculated using a discounted cash flow (DCF) model. This model assumes that the company will continue to generate future cash flows at a constant rate, and that the cash flows are discounted to their present value using a discount rate. The discount rate is typically based on the risk-free rate of interest, which is the interest rate that would be charged on a similar investment with the same risk.

The cost of equity is typically expressed as a percentage, and it is calculated by dividing the ROE by the COE. The higher the cost of equity, the more expensive it is to finance the company's operations. A lower cost of equity indicates that the company's investment is more profitable and may be more attractive to investors.

Overall, the cost of equity is an important financial metric for investors to consider when evaluating the risk and return of a company's investment. It provides a measure of the cost of capital and helps investors make informed decisions about investing in a company.
```

The Explanation Video can be found by clicking the link [Intel Project Video](https://drive.google.com/file/d/1y0zHbecpfFLH25lg24_Wu-sAX7mu-kjY/view?usp=sharing).