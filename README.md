# CS467 Project

### Getting started
First, we should create the Python virtual environment. This will keep our project's installed dependencies/libraries separate from our system's.  
Next we want to go ahead and install those dependencies in the virtual environment.  
Finally, we can run the code and calculate the outputs which will then be stored in the outputs folder one level up from the project folder.

#### Step-by-step example for the first run
##### Windows
###### cmd.exe
1. <code>python -m venv venv</code>
2. <code>venv\Scripts\activate.bat</code>
3. <code>pip install -r requirements.txt</code>
4. <code>python project_467_v03.py</code>
5. Open security_list.txt and add symbols line by line into the file.
6. <code>python project_467_v03.py</code>
7. Calculation results can then be found in the outputs folder.

###### PowerShell
1. <code>python -m venv venv</code>
2. <code>venv\Scripts\Activate.ps1</code>
3. <code>pip install -r requirements.txt</code>
4. <code>python project_467_v03.py</code>
5. Open security_list.txt and add symbols line by line into the file.
6. <code>python project_467_v03.py</code>
7. Calculation results can then be found in the outputs folder.

##### Mac/Linux
1. <code>python -m venv venv</code>
2. <code>source venv/bin/activate</code>
3. <code>pip install -r requirements.txt</code>
4. <code>python project_467_v03.py</code>
5. Open security_list.txt and add symbols line by line into the file.
6. <code>python project_467_v03.py</code>
7. Calculation results can then be found in the outputs folder.

### data_retrieval.py
#### Using get_history_yf to create a csv containing VTI's historical data
1. Create your Python virtual environment, activate it, and then install the requirements with pip. (Exact steps can be found [above](#step-by-step-example-for-the-first-run) in steps 1 to 3.)
2. Create a file with the following content:
```python
from data_retrieval import get_history_yf

df = get_history_yf("VTI")
df.to_csv("VTI.csv")  # Specify path to write csv here
```
3. Run the Python file that was created above. A csv file named <code>VTI.csv</code> should now have been created with 5 minute data of the past 60 days.

#### Using get_history_apca to create a csv containing VTI's historical data
1. Go to https://alpaca.markets/
   1. Sign up
   2. Log in
   3. View your API keys on the righthand side
   4. Generate new key
   5. Note down your API Key ID and Secret Key for step 3, do not share this information with anyone you do not trust
2. Create your Python virtual environment, activate it, and then install the requirements with pip. (Exact steps can be found [above](#step-by-step-example-for-the-first-run) in steps 1 to 3.)
3. Create a file named <code>.env</code>. This is where the key details from step 1.5 come into play. Make sure <code>.env</code> is in the <code>.gitignore</code> if you plan on committing anything. The file should look like the following:
```
APCA_API_KEY_ID=<replace the angle brackets and text within>
APCA_API_SECRET_KEY=<replace the angle brackets and text within>
```
4. Create a file with the following content:
```python
from data_retrieval import get_history_apca

df = get_history_apca("VTI")
df.to_csv("VTI.csv")  # Specify path to write csv here
```
5. Run the Python file that was created above. A csv file named <code>VTI.csv</code> should now have been created with 5 minute data of the past few years.

NOTE: This function only gets data up to yesterday from the IEX (Investors Exchange LLC) which accounts for ~2.5% market volume. It's unable to get data for the current day.
