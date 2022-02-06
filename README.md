# CS467 Project

### Getting started
First, we should create the Python virtual environment. This will keep our project's installed dependencies/libraries separate from our system's.  
Next we want to go ahead and install those dependencies in the virtual environment.  
Finally, we can run the code and calculate the outputs which will then be stored in the outputs folder one level up from the project folder.

#### Step-by-step example
##### Windows
###### cmd.exe
1. <code>python -m venv venv</code>
2. <code>venv\Scripts\activate.bat</code>
3. <code>pip install -r requirements.txt</code>
4. <code>python project_467_v03.py</code>

###### PowerShell
1. <code>python -m venv venv</code>
2. <code>venv\Scripts\Activate.ps1</code>
3. <code>pip install -r requirements.txt</code>
4. <code>python project_467_v03.py</code>

##### Mac/Linux
1. <code>python -m venv venv</code>
2. <code>source venv/bin/activate</code>
3. <code>pip install -r requirements.txt</code>
4. <code>python project_467_v03.py</code>
