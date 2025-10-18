#include "rwkv_opencog.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <memory>
#include <mutex>
#include <cmath>
#include <algorithm>
#include <cstring>

// Internal atom representation
struct rwkv_atom {
    rwkv_atom_handle_t handle;
    rwkv_atom_type_t type;
    std::string name;  // For nodes
    std::vector<rwkv_atom_handle_t> outgoing;  // For links
    rwkv_truth_value_t tv;
    rwkv_attention_value_t av;
    
    rwkv_atom(rwkv_atom_handle_t h, rwkv_atom_type_t t) 
        : handle(h), type(t), tv{0.5f, 0.1f}, av{0.0f, 0.0f, 0.0f} {}
};

// AtomSpace implementation
struct rwkv_atomspace {
    std::unordered_map<rwkv_atom_handle_t, std::unique_ptr<rwkv_atom>> atoms;
    std::unordered_map<std::string, std::vector<rwkv_atom_handle_t>> name_index;  // For node lookup
    std::unordered_set<std::string> link_signatures;  // For link deduplication
    rwkv_atom_handle_t next_handle;
    std::mutex mutex;  // Thread safety
    
    rwkv_atomspace() : next_handle(1) {}
};

// Utility functions
static std::string create_link_signature(rwkv_atom_type_t type, const rwkv_atom_handle_t * outgoing, size_t count) {
    std::string sig = std::to_string(type) + ":";
    for (size_t i = 0; i < count; i++) {
        if (i > 0) sig += ",";
        sig += std::to_string(outgoing[i]);
    }
    return sig;
}

static bool is_node_type(rwkv_atom_type_t type) {
    return type == RWKV_ATOM_NODE || 
           type == RWKV_ATOM_CONCEPT_NODE || 
           type == RWKV_ATOM_PREDICATE_NODE ||
           type == RWKV_ATOM_NUMBER_NODE ||
           type == RWKV_ATOM_VARIABLE_NODE;
}

static bool is_link_type(rwkv_atom_type_t type) {
    return !is_node_type(type);
}

// AtomSpace creation and destruction
struct rwkv_atomspace * rwkv_atomspace_create(void) {
    try {
        return new rwkv_atomspace();
    } catch (...) {
        return nullptr;
    }
}

void rwkv_atomspace_free(struct rwkv_atomspace * atomspace) {
    delete atomspace;
}

// Atom creation
rwkv_atom_handle_t rwkv_atomspace_add_node(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_type_t type,
    const char * name
) {
    if (!atomspace || !name || !is_node_type(type)) {
        return RWKV_INVALID_ATOM_HANDLE;
    }
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    
    std::string node_name(name);
    
    // Check if node already exists
    auto it = atomspace->name_index.find(node_name);
    if (it != atomspace->name_index.end()) {
        for (rwkv_atom_handle_t handle : it->second) {
            auto atom_it = atomspace->atoms.find(handle);
            if (atom_it != atomspace->atoms.end() && atom_it->second->type == type) {
                return handle;  // Return existing atom
            }
        }
    }
    
    // Create new atom
    rwkv_atom_handle_t handle = atomspace->next_handle++;
    std::unique_ptr<rwkv_atom> atom(new rwkv_atom(handle, type));
    atom->name = node_name;
    
    atomspace->atoms[handle] = std::move(atom);
    atomspace->name_index[node_name].push_back(handle);
    
    return handle;
}

rwkv_atom_handle_t rwkv_atomspace_add_link(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_type_t type,
    const rwkv_atom_handle_t * outgoing,
    size_t outgoing_count
) {
    if (!atomspace || !outgoing || outgoing_count == 0 || !is_link_type(type)) {
        return RWKV_INVALID_ATOM_HANDLE;
    }
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    
    // Verify all outgoing atoms exist
    for (size_t i = 0; i < outgoing_count; i++) {
        if (atomspace->atoms.find(outgoing[i]) == atomspace->atoms.end()) {
            return RWKV_INVALID_ATOM_HANDLE;
        }
    }
    
    // Check if link already exists
    std::string signature = create_link_signature(type, outgoing, outgoing_count);
    if (atomspace->link_signatures.find(signature) != atomspace->link_signatures.end()) {
        // Find and return existing link handle
        for (auto it = atomspace->atoms.begin(); it != atomspace->atoms.end(); ++it) {
            rwkv_atom_handle_t handle = it->first;
            const std::unique_ptr<rwkv_atom>& atom = it->second;
            if (atom->type == type && atom->outgoing.size() == outgoing_count) {
                bool match = true;
                for (size_t i = 0; i < outgoing_count; i++) {
                    if (atom->outgoing[i] != outgoing[i]) {
                        match = false;
                        break;
                    }
                }
                if (match) return handle;
            }
        }
    }
    
    // Create new link
    rwkv_atom_handle_t handle = atomspace->next_handle++;
    std::unique_ptr<rwkv_atom> atom(new rwkv_atom(handle, type));
    atom->outgoing.assign(outgoing, outgoing + outgoing_count);
    
    atomspace->atoms[handle] = std::move(atom);
    atomspace->link_signatures.insert(signature);
    
    return handle;
}

// Atom retrieval
struct rwkv_atom * rwkv_atomspace_get_atom(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t handle
) {
    if (!atomspace || handle == RWKV_INVALID_ATOM_HANDLE) {
        return nullptr;
    }
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    auto it = atomspace->atoms.find(handle);
    return (it != atomspace->atoms.end()) ? it->second.get() : nullptr;
}

// Truth value operations
bool rwkv_atom_set_truth_value(
    struct rwkv_atom * atom,
    const rwkv_truth_value_t * tv
) {
    if (!atom || !tv) return false;
    
    atom->tv = *tv;
    // Clamp values to valid ranges
    atom->tv.strength = std::max(0.0f, std::min(1.0f, atom->tv.strength));
    atom->tv.confidence = std::max(0.0f, std::min(1.0f, atom->tv.confidence));
    return true;
}

bool rwkv_atom_get_truth_value(
    struct rwkv_atom * atom,
    rwkv_truth_value_t * tv
) {
    if (!atom || !tv) return false;
    *tv = atom->tv;
    return true;
}

// Attention value operations
bool rwkv_atom_set_attention_value(
    struct rwkv_atom * atom,
    const rwkv_attention_value_t * av
) {
    if (!atom || !av) return false;
    atom->av = *av;
    return true;
}

bool rwkv_atom_get_attention_value(
    struct rwkv_atom * atom,
    rwkv_attention_value_t * av
) {
    if (!atom || !av) return false;
    *av = atom->av;
    return true;
}

// Atom property access
rwkv_atom_type_t rwkv_atom_get_type(struct rwkv_atom * atom) {
    return atom ? atom->type : RWKV_ATOM_NODE;
}

const char * rwkv_atom_get_name(struct rwkv_atom * atom) {
    if (!atom || is_link_type(atom->type)) return nullptr;
    return atom->name.c_str();
}

size_t rwkv_atom_get_outgoing(
    struct rwkv_atom * atom,
    rwkv_atom_handle_t * outgoing,
    size_t max_count
) {
    if (!atom || !outgoing || !is_link_type(atom->type)) return 0;
    
    size_t count = std::min(max_count, atom->outgoing.size());
    for (size_t i = 0; i < count; i++) {
        outgoing[i] = atom->outgoing[i];
    }
    return count;
}

// Pattern matching (simplified implementation)
size_t rwkv_atomspace_pattern_match(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t pattern,
    rwkv_atom_handle_t * results,
    size_t max_results
) {
    if (!atomspace || pattern == RWKV_INVALID_ATOM_HANDLE || !results) return 0;
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    
    auto pattern_atom = atomspace->atoms.find(pattern);
    if (pattern_atom == atomspace->atoms.end()) return 0;
    
    size_t count = 0;
    
    // Simple pattern matching: find atoms of the same type
    for (auto it = atomspace->atoms.begin(); it != atomspace->atoms.end(); ++it) {
        if (count >= max_results) break;
        
        rwkv_atom_handle_t handle = it->first;
        const std::unique_ptr<rwkv_atom>& atom = it->second;
        
        if (atom->type == pattern_atom->second->type && handle != pattern) {
            results[count++] = handle;
        }
    }
    
    return count;
}

// RWKV integration functions
bool rwkv_context_to_atoms(
    struct rwkv_context * rwkv_ctx,
    struct rwkv_atomspace * atomspace,
    const float * state,
    size_t state_len
) {
    // rwkv_ctx can be NULL for standalone AtomSpace usage
    if (!atomspace || !state) return false;
    
    // Convert significant state values to concept nodes
    // This is a simplified implementation - in practice, you'd want more sophisticated conversion
    
    for (size_t i = 0; i < std::min(state_len, size_t(100)); i++) {
        if (std::abs(state[i]) > 0.1f) {  // Only consider significant activations
            std::string concept_name = "state_" + std::to_string(i);
            rwkv_atom_handle_t handle = rwkv_atomspace_add_node(
                atomspace, RWKV_ATOM_CONCEPT_NODE, concept_name.c_str()
            );
            
            if (handle != RWKV_INVALID_ATOM_HANDLE) {
                struct rwkv_atom * atom = rwkv_atomspace_get_atom(atomspace, handle);
                if (atom) {
                    rwkv_truth_value_t tv = {std::abs(state[i]), 0.8f};
                    rwkv_atom_set_truth_value(atom, &tv);
                    
                    rwkv_attention_value_t av = {state[i] > 0 ? state[i] : 0.0f, 0.0f, 0.0f};
                    rwkv_atom_set_attention_value(atom, &av);
                }
            }
        }
    }
    
    return true;
}

bool rwkv_atoms_to_context(
    struct rwkv_atomspace * atomspace,
    struct rwkv_context * rwkv_ctx,
    float * state,
    size_t state_len
) {
    // rwkv_ctx can be NULL for standalone AtomSpace usage
    if (!atomspace || !state) return false;
    
    // Initialize state to zeros
    memset(state, 0, state_len * sizeof(float));
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    
    // Convert atoms back to state values
    // This is a simplified implementation
    for (auto it = atomspace->atoms.begin(); it != atomspace->atoms.end(); ++it) {
        const std::unique_ptr<rwkv_atom>& atom = it->second;
        if (atom->type == RWKV_ATOM_CONCEPT_NODE && atom->name.find("state_") == 0) {
            try {
                size_t index = std::stoul(atom->name.substr(6));  // Remove "state_" prefix
                if (index < state_len) {
                    state[index] = atom->tv.strength * (atom->av.sti > 0 ? 1.0f : -1.0f);
                }
            } catch (...) {
                // Ignore parsing errors
            }
        }
    }
    
    return true;
}

// Simple forward inference
bool rwkv_atomspace_forward_inference(
    struct rwkv_atomspace * atomspace,
    rwkv_atom_handle_t premise,
    rwkv_atom_handle_t * conclusions,
    size_t max_conclusions,
    size_t * num_conclusions
) {
    if (!atomspace || premise == RWKV_INVALID_ATOM_HANDLE || !conclusions || !num_conclusions) {
        return false;
    }
    
    *num_conclusions = 0;
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    
    // Simple inference: if we have an implication link A -> B and we know A, infer B
    for (auto it = atomspace->atoms.begin(); it != atomspace->atoms.end(); ++it) {
        if (*num_conclusions >= max_conclusions) break;
        
        const std::unique_ptr<rwkv_atom>& atom = it->second;
        if (atom->type == RWKV_ATOM_IMPLICATION_LINK && atom->outgoing.size() == 2) {
            if (atom->outgoing[0] == premise) {
                conclusions[(*num_conclusions)++] = atom->outgoing[1];
            }
        }
    }
    
    return true;
}

// Memory consolidation
bool rwkv_atomspace_consolidate_memory(
    struct rwkv_atomspace * atomspace,
    float /* similarity_threshold */
) {
    if (!atomspace) return false;
    
    // This is a placeholder for memory consolidation
    // In a full implementation, you would:
    // 1. Find similar atoms based on truth values and attention
    // 2. Merge atoms that are above the similarity threshold
    // 3. Update links that reference merged atoms
    
    return true;
}

// Statistics
size_t rwkv_atomspace_get_size(struct rwkv_atomspace * atomspace) {
    if (!atomspace) return 0;
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    return atomspace->atoms.size();
}

size_t rwkv_atomspace_get_node_count(struct rwkv_atomspace * atomspace) {
    if (!atomspace) return 0;
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    size_t count = 0;
    for (auto it = atomspace->atoms.begin(); it != atomspace->atoms.end(); ++it) {
        const std::unique_ptr<rwkv_atom>& atom = it->second;
        if (is_node_type(atom->type)) count++;
    }
    return count;
}

size_t rwkv_atomspace_get_link_count(struct rwkv_atomspace * atomspace) {
    if (!atomspace) return 0;
    
    std::lock_guard<std::mutex> lock(atomspace->mutex);
    size_t count = 0;
    for (auto it = atomspace->atoms.begin(); it != atomspace->atoms.end(); ++it) {
        const std::unique_ptr<rwkv_atom>& atom = it->second;
        if (is_link_type(atom->type)) count++;
    }
    return count;
}