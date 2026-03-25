import json
import jsonschema
from jsonschema import Draft7Validator

def validate_and_repair_json(schema_data, input_json_path, output_json_path, log_path):
    # Load input data
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    validator = Draft7Validator(schema_data)
    errors = list(validator.iter_errors(data))
    
    error_log = []
    
    for error in errors:
        # Record error details
        path_list = list(error.path)
        error_log.append({
            "field_path": " -> ".join(map(str, path_list)) if path_list else "root",
            "error_message": error.message,
            "validator": error.validator,
            "invalid_value": error.instance
        })

        # Repair: Traverse to the invalid field and set to None
        if path_list:
            target = data
            for key in path_list[:-1]:
                target = target[key]
            
            # Nullify the field causing the specific validation error
            target[path_list[-1]] = None

    # Write corrected JSON
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    # Write error log
    with open(log_path, 'w') as f:
        json.dump(error_log, f, indent=4)

    return len(error_log)

# Usage Example