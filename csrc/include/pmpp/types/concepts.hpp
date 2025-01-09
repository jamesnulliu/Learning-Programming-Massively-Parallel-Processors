#pragma once

#include <iterator>
#include <ranges>

namespace pmpp
{

template <typename T>
concept is_readable_range = requires(T& t) {
    typename std::ranges::range_value_t<T>;
    { std::begin(t) } -> std::input_iterator;
    { std::end(t) } -> std::sentinel_for<std::ranges::iterator_t<T>>;
};

template <typename T>
concept is_range = is_readable_range<T>;

template <typename T>
concept is_writable_range = is_readable_range<T> && requires(T& t) {
    { std::begin(t) } -> std::output_iterator<std::ranges::range_value_t<T>>;
};

template <typename T>
struct GetInnerTypes;

template <template <typename...> class TemplateClass, typename... InnerTypes>
struct GetInnerTypes<TemplateClass<InnerTypes...>>
{
    using inner_types = std::tuple<InnerTypes...>;
};

template <typename T>
using GetInnerTypes_t = typename GetInnerTypes<T>::types;

template <typename T, std::size_t N = 0>
using GetInnerType_t = std::tuple_element_t<N, GetInnerTypes_t<T>>;

}  // namespace pmpp