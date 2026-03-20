import re


def apply_twikit_patches():
    try:
        import twikit.x_client_transaction.transaction as tx
    except Exception:
        return False

    if getattr(tx, "_IKAIMO_ONDEMAND_PATCHED", False):
        return True

    tx.ON_DEMAND_FILE_REGEX = re.compile(
        r""",(\d+):["']ondemand\.s["']""",
        flags=(re.VERBOSE | re.MULTILINE),
    )
    tx.ON_DEMAND_HASH_PATTERN = r""",{}:["']([0-9a-f]+)["']"""

    async def patched_get_indices(self, home_page_response, session, headers):
        key_byte_indices = []
        response = self.validate_response(home_page_response) or self.home_page_response
        response_text = str(response or "")

        on_demand_file_index_match = tx.ON_DEMAND_FILE_REGEX.search(response_text)
        if not on_demand_file_index_match:
            raise Exception("Couldn't resolve ondemand file index")

        on_demand_file_index = on_demand_file_index_match.group(1)
        filename_regex = re.compile(tx.ON_DEMAND_HASH_PATTERN.format(on_demand_file_index))
        filename_match = filename_regex.search(response_text)
        if not filename_match:
            raise Exception("Couldn't resolve ondemand file hash")

        filename = filename_match.group(1)
        on_demand_file_url = (
            "https://abs.twimg.com/responsive-web/client-web/"
            f"ondemand.s.{filename}a.js"
        )
        on_demand_file_response = await session.request(method="GET", url=on_demand_file_url, headers=headers)
        key_byte_indices_match = tx.INDICES_REGEX.finditer(str(on_demand_file_response.text))
        for item in key_byte_indices_match:
            key_byte_indices.append(item.group(2))
        if not key_byte_indices:
            raise Exception("Couldn't get KEY_BYTE indices")
        key_byte_indices = list(map(int, key_byte_indices))
        return key_byte_indices[0], key_byte_indices[1:]

    tx.ClientTransaction.get_indices = patched_get_indices
    tx._IKAIMO_ONDEMAND_PATCHED = True
    return True
