#include "template_project_name/system.hpp"

namespace template_project_name::cuda
{
VSC_PYTHON_TEMPLATE_API void launch_vec_add(const float* a, const float* b,
                                            float* c, int n);
}

namespace template_project_name::cpu
{
VSC_PYTHON_TEMPLATE_API void launch_vec_add(const float* a, const float* b,
                                            float* c, int n);
}
