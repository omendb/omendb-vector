/*
 * filter_fast.c — Fast metadata filter evaluation in C
 *
 * Takes metadata as JSON strings + filter as JSON string.
 * Returns matching indices.
 *
 * Uses cJSON for JSON parsing (already vendored).
 */

#include <stdlib.h>
#include <string.h>
#include "cJSON.h"

/* Filter operators */
#define OP_EQ   0
#define OP_GT   1
#define OP_LT   2
#define OP_GTE  3
#define OP_LTE  4
#define OP_IN   5
#define OP_EXISTS 6

typedef struct {
    const char *key;    /* points into filter JSON, don't free */
    int op;
    double num_val;
    const char *str_val; /* points into filter JSON, don't free */
    cJSON *array_val;    /* points into filter JSON, don't free */
    int exists_val;
} FilterCondition;

/*
 * Parse filter JSON into conditions array.
 * Returns number of conditions, or -1 on error.
 */
static int parse_filter(const char *filter_json, FilterCondition *conditions, int max_conditions) {
    cJSON *filter = cJSON_Parse(filter_json);
    if (!filter) return -1;

    int count = 0;
    cJSON *item = NULL;

    cJSON_ArrayForEach(item, filter) {
        if (count >= max_conditions) break;

        const char *key = item->string;

        if (cJSON_IsObject(item)) {
            /* Operator filter: {"score": {"$gt": 50}} */
            cJSON *op_item = NULL;
            cJSON_ArrayForEach(op_item, item) {
                if (count >= max_conditions) break;

                conditions[count].key = key;
                conditions[count].str_val = NULL;
                conditions[count].array_val = NULL;
                conditions[count].exists_val = 0;

                const char *op_str = op_item->string;
                if (strcmp(op_str, "$gt") == 0) {
                    conditions[count].op = OP_GT;
                    conditions[count].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$lt") == 0) {
                    conditions[count].op = OP_LT;
                    conditions[count].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$gte") == 0) {
                    conditions[count].op = OP_GTE;
                    conditions[count].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$lte") == 0) {
                    conditions[count].op = OP_LTE;
                    conditions[count].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$in") == 0) {
                    conditions[count].op = OP_IN;
                    conditions[count].array_val = op_item;
                } else if (strcmp(op_str, "$exists") == 0) {
                    conditions[count].op = OP_EXISTS;
                    conditions[count].exists_val = cJSON_IsTrue(op_item) ? 1 : 0;
                }
                count++;
            }
        } else {
            /* Equality filter: {"status": "active"} */
            conditions[count].key = key;
            conditions[count].op = OP_EQ;
            conditions[count].str_val = NULL;
            conditions[count].array_val = NULL;
            conditions[count].exists_val = 0;
            if (cJSON_IsString(item)) {
                conditions[count].str_val = item->valuestring;
            } else if (cJSON_IsNumber(item)) {
                conditions[count].num_val = cJSON_GetNumberValue(item);
            }
            count++;
        }
    }

    /* Note: filter cJSON is NOT freed here because conditions point into it.
     * Caller must keep filter_json alive until after matches_conditions calls.
     * We'll parse filter_json in the main function and free there. */
    cJSON_Delete(filter);

    return count;
}

/*
 * Check if a metadata object matches all conditions.
 */
static int matches_conditions(cJSON *metadata, FilterCondition *conditions, int num_conditions) {
    if (!metadata && num_conditions > 0) return 0;

    for (int i = 0; i < num_conditions; i++) {
        cJSON *field = cJSON_GetObjectItemCaseSensitive(metadata, conditions[i].key);
        int exists = (field != NULL);

        switch (conditions[i].op) {
            case OP_EXISTS:
                if (exists != conditions[i].exists_val) return 0;
                break;

            case OP_EQ:
                if (!exists) return 0;
                if (conditions[i].str_val) {
                    if (!cJSON_IsString(field) || strcmp(field->valuestring, conditions[i].str_val) != 0)
                        return 0;
                } else {
                    if (!cJSON_IsNumber(field) || field->valuedouble != conditions[i].num_val)
                        return 0;
                }
                break;

            case OP_GT:
                if (!exists || !cJSON_IsNumber(field) || field->valuedouble <= conditions[i].num_val)
                    return 0;
                break;

            case OP_LT:
                if (!exists || !cJSON_IsNumber(field) || field->valuedouble >= conditions[i].num_val)
                    return 0;
                break;

            case OP_GTE:
                if (!exists || !cJSON_IsNumber(field) || field->valuedouble < conditions[i].num_val)
                    return 0;
                break;

            case OP_LTE:
                if (!exists || !cJSON_IsNumber(field) || field->valuedouble > conditions[i].num_val)
                    return 0;
                break;

            case OP_IN:
                if (!exists) return 0;
                if (!cJSON_IsArray(conditions[i].array_val)) return 0;
                {
                    int found = 0;
                    cJSON *element = NULL;
                    cJSON_ArrayForEach(element, conditions[i].array_val) {
                        if (cJSON_IsNumber(element) && cJSON_IsNumber(field) &&
                            element->valuedouble == field->valuedouble) {
                            found = 1;
                            break;
                        }
                        if (cJSON_IsString(element) && cJSON_IsString(field) &&
                            strcmp(element->valuestring, field->valuestring) == 0) {
                            found = 1;
                            break;
                        }
                    }
                    if (!found) return 0;
                }
                break;
        }
    }
    return 1;
}

/*
 * Main entry point: filter metadata and return matching indices.
 *
 * Returns: number of matching indices, or -1 on error.
 */
int filter_batch_fast(
    const char **metadata_jsons,
    int num_items,
    const int *deleted,
    const char *filter_json,
    int *out_indices,
    int max_out
) {
    /* Parse filter once */
    cJSON *filter_root = cJSON_Parse(filter_json);
    if (!filter_root) return -1;

    /* Build conditions array from parsed filter */
    FilterCondition conditions[32];
    int num_conditions = 0;

    cJSON *item = NULL;
    cJSON_ArrayForEach(item, filter_root) {
        if (num_conditions >= 32) break;

        const char *key = item->string;

        if (cJSON_IsObject(item)) {
            cJSON *op_item = NULL;
            cJSON_ArrayForEach(op_item, item) {
                if (num_conditions >= 32) break;

                conditions[num_conditions].key = key;
                conditions[num_conditions].str_val = NULL;
                conditions[num_conditions].array_val = NULL;
                conditions[num_conditions].exists_val = 0;

                const char *op_str = op_item->string;
                if (strcmp(op_str, "$gt") == 0) {
                    conditions[num_conditions].op = OP_GT;
                    conditions[num_conditions].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$lt") == 0) {
                    conditions[num_conditions].op = OP_LT;
                    conditions[num_conditions].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$gte") == 0) {
                    conditions[num_conditions].op = OP_GTE;
                    conditions[num_conditions].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$lte") == 0) {
                    conditions[num_conditions].op = OP_LTE;
                    conditions[num_conditions].num_val = cJSON_GetNumberValue(op_item);
                } else if (strcmp(op_str, "$in") == 0) {
                    conditions[num_conditions].op = OP_IN;
                    conditions[num_conditions].array_val = op_item;
                } else if (strcmp(op_str, "$exists") == 0) {
                    conditions[num_conditions].op = OP_EXISTS;
                    conditions[num_conditions].exists_val = cJSON_IsTrue(op_item) ? 1 : 0;
                }
                num_conditions++;
            }
        } else {
            conditions[num_conditions].key = key;
            conditions[num_conditions].op = OP_EQ;
            conditions[num_conditions].str_val = NULL;
            conditions[num_conditions].array_val = NULL;
            conditions[num_conditions].exists_val = 0;
            if (cJSON_IsString(item)) {
                conditions[num_conditions].str_val = item->valuestring;
            } else if (cJSON_IsNumber(item)) {
                conditions[num_conditions].num_val = cJSON_GetNumberValue(item);
            }
            num_conditions++;
        }
    }

    /* Evaluate filter on each item */
    int out_count = 0;
    for (int i = 0; i < num_items && out_count < max_out; i++) {
        if (deleted[i]) continue;

        const char *meta_str = metadata_jsons[i];
        if (!meta_str) {
            /* No metadata — only $exists:false matches */
            int match = 1;
            for (int j = 0; j < num_conditions; j++) {
                if (conditions[j].op != OP_EXISTS || conditions[j].exists_val != 0) {
                    match = 0;
                    break;
                }
            }
            if (match) out_indices[out_count++] = i;
            continue;
        }

        cJSON *metadata = cJSON_Parse(meta_str);
        if (!metadata) continue;

        if (matches_conditions(metadata, conditions, num_conditions)) {
            out_indices[out_count++] = i;
        }

        cJSON_Delete(metadata);
    }

    cJSON_Delete(filter_root);
    return out_count;
}
