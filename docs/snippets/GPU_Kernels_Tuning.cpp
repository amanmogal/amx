#include <ie_core.hpp>
#include "cldnn/cldnn_config.hpp"

int main() {
using namespace InferenceEngine;
//! [part0]
Core ie;          
  ie.SetConfig({{ CONFIG_KEY(TUNING_MODE), CONFIG_VALUE(TUNING_CREATE) }}, "GPU");
  ie.SetConfig({{ CONFIG_KEY(TUNING_FILE), "/path/to/tuning/file.json" }}, "GPU");
  // Further LoadNetwork calls will use the specified tuning parameters
//! [part0]

return 0;
}
