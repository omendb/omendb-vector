/* filter_batch.c - Batch filter evaluation using cJSON.
 *
 * Single C call that iterates all items, parses metadata with cJSON,
 * evaluates filter predicates, and returns matching indices.
 * No per-item Python/ctypes overhead.
 */

#include "cJSON.h"
#include <string.h>
#include <stdlib.h>

static int value_in_array(const cJSON *value, const cJSON *array) {
    if (!cJSON_IsArray(array) || !value) return 0;
    int size = cJSON_GetArraySize(array);
    for (int i = 0; i < size; i++) {
        cJSON *item = cJSON_GetArrayItem(array, i);
        if (cJSON_IsNumber(item) && cJSON_IsNumber(value)) {
            if (cJSON_GetNumberValue(item) == cJSON_GetNumberValue(value))
                return 1;
        } else if (cJSON_IsString(item) && cJSON_IsString(value)) {
            char *a = cJSON_GetStringValue(item);
            char *b = cJSON_GetStringValue(value);
            if (a && b && strcmp(a, b) == 0)
                return 1;
        }
    }
    return 0;
}

static int eval_condition(const cJSON *actual, const char *op, const cJSON *expected) {
    if (strcmp(op, "$exists") == 0) {
        int exists = (actual != NULL);
        int want = cJSON_IsTrue(expected) ? 1 : 0;
        return exists == want;
    }
    if (actual == NULL) return 0;

    if (strcmp(op, "") == 0) {
        if (cJSON_IsNumber(actual) && cJSON_IsNumber(expected))
            return cJSON_GetNumberValue(actual) == cJSON_GetNumberValue(expected);
        if (cJSON_IsString(actual) && cJSON_IsString(expected)) {
            char *a = cJSON_GetStringValue(actual);
            char *b = cJSON_GetStringValue(expected);
            return (a && b && strcmp(a, b) == 0);
        }
        if (cJSON_IsBool(actual) && cJSON_IsBool(expected))
            return cJSON_IsTrue(actual) == cJSON_IsTrue(expected);
        return 0;
    }

    if (strcmp(op, "$in") == 0)
        return value_in_array(actual, expected);

    if (!cJSON_IsNumber(actual)) return 0;
    double val = cJSON_GetNumberValue(actual);
    double threshold = cJSON_GetNumberValue(expected);

    if (strcmp(op, "$gt") == 0)  return val > threshold;
    if (strcmp(op, "$lt") == 0)  return val < threshold;
    if (strcmp(op, "$gte") == 0) return val >= threshold;
    if (strcmp(op, "$lte") == 0) return val <= threshold;

    return -1;
}

static int matches_filter(const char *metadata_json, cJSON *filter) {
    cJSON *metadata = cJSON_Parse(metadata_json);
    if (!metadata) {
        /* No valid metadata - check if filter expects non-existence */
        cJSON *fi = NULL;
        int result = 1;
        cJSON_ArrayForEach(fi, filter) {
            if (cJSON_IsObject(fi)) {
                cJSON *exists_op = cJSON_GetObjectItem(fi, "$exists");
                if (exists_op && cJSON_IsTrue(exists_op)) {
                    result = 0;
                    break;
                }
            } else {
                result = 0;
                break;
            }
        }
        if (metadata) cJSON_Delete(metadata);
        return result;
    }

    int result = 1;
    cJSON *fi = NULL;
    cJSON_ArrayForEach(fi, filter) {
        char *key = fi->string;
        if (!key) continue;

        cJSON *actual = cJSON_GetObjectItem(metadata, key);

        if (cJSON_IsObject(fi)) {
            cJSON *op_item = NULL;
            cJSON_ArrayForEach(op_item, fi) {
                char *op = op_item->string;
                if (!op) continue;
                if (eval_condition(actual, op, op_item) != 1) {
                    result = 0;
                    goto done;
                }
            }
        } else {
            if (eval_condition(actual, "", fi) != 1) {
                result = 0;
                goto done;
            }
        }
    }

done:
    cJSON_Delete(metadata);
    return result;
}

/* Batch filter: iterate all items, return matching indices.
 *
 * metadata_strings: array of C strings (JSON metadata per item)
 * deleted: array of ints (1 = deleted, 0 = live)
 * n: number of items
 * filter_json: JSON string describing the filter
 * out_indices: pre-allocated output array for matching indices
 * returns: number of matching items written to out_indices
 */
int filter_batch(
    const char **metadata_strings,
    const int *deleted,
    int n,
    const char *filter_json,
    int *out_indices
) {
    cJSON *filter = cJSON_Parse(filter_json);
    if (!filter) return -1;

    int count = 0;
    for (int i = 0; i < n; i++) {
        if (deleted[i]) continue;
        const char *meta = metadata_strings[i] ? metadata_strings[i] : "{}";
        if (matches_filter(meta, filter)) {
            out_indices[count++] = i;
        }
    }

    cJSON_Delete(filter);
    return count;
}
