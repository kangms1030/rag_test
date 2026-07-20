# Claude Code Settings for Intern Chatbot

## Terminal Commands
1. Every terminal command MUST start with `cmd /c`.
2. For Python or Conda tasks, ALWAYS use this exact sequence:
   ```
   cmd /c conda activate intern_chatbot && <your_command_here>
   ```
3. DO NOT wrap the entire command after `cmd /c` in double quotes unless absolutely necessary, to avoid syntax errors in Windows.

## File Operations
4. For simple file/folder checks or one-off code execution, generate the code in the `tmp` folder to facilitate subsequent cleanup.
