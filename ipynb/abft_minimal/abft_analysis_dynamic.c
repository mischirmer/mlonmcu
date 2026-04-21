#include "iree/modules/abft_analysis/module.h"
#include "iree/vm/dynamic/api.h"

IREE_VM_DYNAMIC_MODULE_EXPORT iree_status_t iree_vm_dynamic_module_create(
    iree_vm_dynamic_module_version_t max_version,
    iree_vm_instance_t* instance,
    iree_host_size_t param_count,
    const iree_string_pair_t* params,
    iree_allocator_t allocator,
    iree_vm_module_t** out_module) {
  (void)param_count;
  (void)params;
  if (max_version < IREE_VM_DYNAMIC_MODULE_VERSION_LATEST) {
    *out_module = NULL;
    return iree_ok_status();
  }
  return iree_abft_analysis_module_create(instance, allocator, out_module);
}
